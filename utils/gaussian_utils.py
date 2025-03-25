import torch
import numpy as np

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

SHS_REST_DIM_TO_DEGREE = {
    0: 0,
    3: 1,
    8: 2,
    15: 3,
}

class GaussianTransformUtils:
    @staticmethod
    def translation(xyz, x: float, y: float, z: float):
        if x == 0. and y == 0. and z == 0.:
            return xyz

        return xyz + torch.tensor([[x, y, z]], device=xyz.device)

    @staticmethod
    def rescale(xyz, scaling, factor: float):
        if factor == 1.:
            return xyz, scaling
        return xyz * factor, scaling * factor

    @staticmethod
    def rx(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[1, 0, 0],
                             [0, torch.cos(theta), -torch.sin(theta)],
                             [0, torch.sin(theta), torch.cos(theta)]], dtype=torch.float)

    @staticmethod
    def ry(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                             [0, 1, 0],
                             [-torch.sin(theta), 0, torch.cos(theta)]], dtype=torch.float)

    @staticmethod
    def rz(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0],
                             [0, 0, 1]], dtype=torch.float)

    @classmethod
    def rotate_by_euler_angles(cls, xyz, rotation, x: float, y: float, z: float):
        """
        rotate in z-y-x order, radians as unit
        """

        if x == 0. and y == 0. and z == 0.:
            return

        # rotate
        rotation_matrix = cls.rx(x) @ cls.ry(y) @ cls.rz(z)
        xyz, rotation = cls.rotate_by_matrix(
            xyz,
            rotation,
            rotation_matrix.to(xyz),
        )

        return xyz, rotation

    @staticmethod
    def transform_shs(features, rotation_matrix):
        """
        https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
        """

        try:
            from e3nn import o3
            import einops
            from einops import einsum
        except:
            print("Please run `pip install e3nn einops` to enable SHs rotation")
            return features

        if features.shape[1] == 1:
            return features

        features = features.clone()

        shs_feat = features[:, 1:, :]

        ## rotate shs
        P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
        inversed_P = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=shs_feat.dtype, device=shs_feat.device)
        permuted_rotation_matrix = inversed_P @ rotation_matrix @ P
        rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

        # Construction coefficient
        D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
        D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
        D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

        # rotation of the shs features
        one_degree_shs = shs_feat[:, 0:3]
        one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        one_degree_shs = einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
        one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 0:3] = one_degree_shs

        if shs_feat.shape[1] >= 4:
            two_degree_shs = shs_feat[:, 3:8]
            two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            two_degree_shs = einsum(
                D_2,
                two_degree_shs,
                "... i j, ... j -> ... i",
            )
            two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 3:8] = two_degree_shs

            if shs_feat.shape[1] >= 9:
                three_degree_shs = shs_feat[:, 8:15]
                three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                three_degree_shs = einsum(
                    D_3,
                    three_degree_shs,
                    "... i j, ... j -> ... i",
                )
                three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 8:15] = three_degree_shs

        return features

    @classmethod
    def rotate_by_wxyz_quaternions(cls, xyz, rotations, features, quaternions: torch.tensor):
        if torch.all(quaternions == 0.) or torch.all(quaternions == torch.tensor(
                [1., 0., 0., 0.],
                dtype=quaternions.dtype,
                device=quaternions.device,
        )):
            return xyz, rotations, features

        rotation_matrix = torch.tensor(qvec2rotmat(quaternions.cpu().numpy()), dtype=torch.float, device=xyz.device)
        xyz = torch.matmul(xyz, rotation_matrix.T)
        rotations = torch.nn.functional.normalize(cls.quat_multiply(
            rotations,
            quaternions,
        ))

        features = cls.transform_shs(features, rotation_matrix)

        return xyz, rotations, features

    @staticmethod
    def quat_multiply(quaternion0, quaternion1):
        w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
        w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
        return torch.concat((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ), dim=-1)

    @classmethod
    def rotate_by_matrix(cls, xyz, rotations, rotation_matrix):
        xyz = torch.matmul(xyz, rotation_matrix.T)

        rotations = torch.nn.functional.normalize(cls.quat_multiply(
            rotations,
            torch.tensor([rotmat2qvec(rotation_matrix.cpu().numpy())]).to(xyz),
        ))

        return xyz, rotations