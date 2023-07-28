import numpy as np
from numpy.linalg import norm, eig
import sys


class HaloOrientation():

    @staticmethod
    def rotate(v, thetax, thetay, thetaz):
        """Rotate vectors in three dimensions

        Arguments:
            v: vector or set of vectors with dimension (n, 3), where n is the
                number of vectors
            thetax: angle between 0 and 180 degrees
            thetay: angle between 0 and 180 degrees
            thetaz: angle between 0 and 180 degrees

        Returns:
            Rotated vector or set of vectors with dimension (n, 3), where n is the
            number of vectors
        """

        v_new = np.zeros(np.shape(v))
        thetax = thetax * np.pi / 180
        thetay = thetay * np.pi / 180
        thetaz = thetaz * np.pi / 180

        Rx = np.matrix(
            [
                [1, 0, 0],
                [0, np.cos(thetax), -np.sin(thetax)],
                [0, np.sin(thetax), np.cos(thetax)],
            ]
        )

        Ry = np.matrix(
            [
                [np.cos(thetay), 0, np.sin(thetay)],
                [0, 1, 0],
                [-np.sin(thetay), 0, np.cos(thetay)],
            ]
        )

        Rz = np.matrix(
            [
                [np.cos(thetaz), -np.sin(thetaz), 0],
                [np.sin(thetaz), np.cos(thetaz), 0],
                [0, 0, 1],
            ]
        )

        R = Rx * Ry * Rz
        v_new += R * v

        return v_new

    @staticmethod
    def transform(v1, v2):
        """transform to different coordinate system

        Arguments:
            v1: vector or set of vectors with dimension (n, 3), where n is the
                number of vectors
            v2: principal axis of desired coordinate system 

        Returns:
            Vector or set of vectors in new coordinate system with dimension
            (n, 3), where n is the number of vectors
        """

        # Take transpose so that v1[0],v1[1],v1[2] are all x,y,z respectively
        v1 = v1.T
        v_new = np.zeros(np.shape(v1))

        # loop over each of the 3 coordinates
        for i in range(3):
            v_new[i] += v1[0] * v2[i, 0] + v1[1] * v2[i, 1] + v1[2] * v2[i, 2]
            
        return v_new

    @staticmethod
    def generate_random_z_axis_rotation():
            """Generate random rotation matrix about the z axis."""

            R = np.eye(3)
            x1 = np.random.rand()
            R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
            R[0, 1] = -np.sin(2 * np.pi * x1)
            R[1, 0] = np.sin(2 * np.pi * x1)

            return R

    @staticmethod
    def uniform_random_rotation(x):
        """Apply a random rotation in 3D, with a distribution uniform over the
        sphere.

        Arguments:
            x: vector or set of vectors with dimension (n, 3), where n is the
                number of vectors

        Returns:
            Array of shape (n, 3) containing the randomly rotated vectors of x,
            about the mean coordinate of x.

        Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
        https://doi.org/10.1016/B978-0-08-050755-2.50034-8
        """

        # There are two random variables in [0, 1) here (naming is same as paper)
        x2 = 2 * np.pi * np.random.rand()
        x3 = np.random.rand()

        # Rotation of all points around x axis using matrix
        R = HaloOrientation.generate_random_z_axis_rotation()
        v = np.array([np.cos(x2) * np.sqrt(x3), np.sin(x2) * np.sqrt(x3), np.sqrt(1 - x3)])
        H = np.eye(3) - (2 * np.outer(v, v))
        M = -(H @ R)
        x = x.reshape((-1, 3))
        mean_coord = np.mean(x, axis=0)

        return ((x - mean_coord) @ M) + mean_coord @ M

    @staticmethod
    def get_eigs(I, rvir):
        """Get eigenvalues and eigenvectors of halo inertia tensor

        Arguments:
            I (array): host halo inertia tensor
            rvir (array): halo virial radius

        Returns:
            array: eigenvalues
            array: eigenvectors
        """
        # return eigenvectors and eigenvalues
        w, v = eig(I)

        # sort in descending order
        odr = np.argsort(-1.0 * w)

        # sqrt of e values = A,B,C where A is the major axis
        w = np.sqrt(w[odr])
        v = v.T[odr]

        # rescale so major axis = radius of original host
        ratio = rvir / w[0]
        w[0] = w[0] * ratio  # this one is 'A'
        w[1] = w[1] * ratio  # B
        w[2] = w[2] * ratio  # C

        return w, v

    @staticmethod
    def check_ortho(e_vect):
        """Check if eigenvectors inertia tensor are orthogonal

        Arguments:
            e_vect (array): 3x3 array of inertia tensor eigenvectors

        """
        
        #define a diagonal matrix of ones
        a = np.zeros((3, 3))
        np.fill_diagonal(a, 1.)

        #take dot product of e_vect and e_vect.T
        #off diagonals are usually 1e-15 so round them to 0.
        m = np.abs(np.round(np.dot(e_vect,e_vect.T),1))

        #check if all elements are equal to identity matrix
        if np.any(a != m):
            print("not orthonormal")
            sys.exit(1)

    @staticmethod
    def get_perp_dist(I, rvir, pos):
        """ Return component of vector perpendicular to major axis
        and its angular separation from major axis

        Arguments:
            I: array representing host inertia tensor
            rvir: virial radius of halo
            pos: position (x,y,z) vector or set of vectors with dimension (n, 3), 
                where n is the number of vectors

        Returns:
            perp: vector perpendicular to major axis
            t: angular separation between major axis and position vector
        """

        #get prinicpal axis (eigenvectors) from host inertia tensor
        #ev are eigenvectors
        ew, ev = HaloOrientation.get_eigs(I, rvir)
        HaloOrientation.check_ortho(ev)

        #transform position vectors to principal axis coordinate system
        new_pos = HaloOrientation.transform(pos, ev).T

        #define major axis
        #can do for semi-major and minor by indexing 1 or 2 instead of 0
        hA = ev[0]
        hA2 = np.repeat(hA,len(new_pos)).reshape(3,len(new_pos)).T

        #calculate angular separation between rotated position vector and major axis
        t = np.arccos(abs((new_pos*hA2).sum(axis=1)/(norm(new_pos,axis=1)*norm(hA))))

        #dot product to find magnitude of component 
        #of position vectors parallel to major axis
        para1 = (new_pos * hA2 / norm(hA)).sum(axis=1)

        #normalized major axis vector
        para2 = (hA / norm(hA)).T

        #parallel vector (magnitude and direction)
        para = np.array((para2[0] * para1, para2[1] * para1, para2[2] * para1))

        #perpendicular component 
        perp = new_pos - para.T

        return perp, t
