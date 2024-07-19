import numpy as np
from scipy.spatial.transform import Rotation as R

class Transform:
    """
        - rotation: A scipy rotation object
        - translation: A 3x1 numpy array
    """
    def __init__(self, rotation, translation):
        self.R = rotation
        self.t = translation

    def __repr__(self):
        return str(self.__as_matrix__())

    """
    Apply this transformation to a vector `v`
    """
    def __mul__(self, v):
        assert v.shape == (3,) or v.shape == (4,)

        if v.shape == (3,):
            x = np.identity(4)
            x[:3, 3] = v
            res = self.__as_matrix__() @ x
            return res[:3, 3]
        else:
            return self.__as_matrix__() @ v

    """
        - rotation: scipy.Rotation
    """
    def rotate(self, rotation):
        rot = np.identity(4)
        rot[:3, :3] = rotation.as_matrix()
        
        T = self.__as_matrix__() @ rot

        # Extract the rotation and translation
        self.R = R.from_matrix(T[:3, :3])
        self.t = T[:3, 3]

        return self

    def translate(self, v):
        t = np.identity(4)
        t[:3, 3] = v
        T = self.__as_matrix__() @ t
        self.__update_iso__(T)
        return self

    def pretranslate(self, v):
        t = np.identity(4)
        t[:3, 3] = v
        T = t @ self.__as_matrix__()
        self.__update_iso__(T)
        return self

    def inv(self):
        return np.linalg.inv(self.__as_matrix__())


    def __as_matrix__(self) -> np.array:
        T = np.identity(4)
        T[:3, :3] = self.R.as_matrix()
        T[:3, 3] = self.t
        return T

    def __update_iso__(self, T):
        # Extract the rotation and translation
        self.R = R.from_matrix(T[:3, :3])
        self.t = T[:3, 3]


    @staticmethod
    def identity():
        return Transform(R.identity(), np.array([0,0,0]))