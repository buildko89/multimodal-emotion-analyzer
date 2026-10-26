"""Integer grayscale coefficients and boundary rounding are a versioned contract."""
import unittest
import numpy as np
from models.fer_preprocessing import preprocess_face


class FERPreprocessingTests(unittest.TestCase):
    def test_primary_colors_match_integer_definition(self):
        face=np.array([[[255,0,0],[0,255,0],[0,0,255],[255,255,255],[0,0,0]]],np.uint8)
        tensor=preprocess_face(face,(5,1))
        expected=np.array([29,150,76,255,0],np.float32)/np.float32(255)
        expected=(expected-np.float32(.5))*np.float32(2)
        self.assertTrue(np.array_equal(tensor[0,0,:,0],expected))

    def test_gray_identity_and_shape(self):
        face=np.arange(16,dtype=np.uint8).reshape(4,4)
        result=preprocess_face(face,(4,4))
        self.assertEqual(result.shape,(1,4,4,1));self.assertEqual(result.dtype,np.float32)
        self.assertAlmostEqual(float(result[0,0,0,0]),-1.)

    def test_reject_float_or_alpha(self):
        with self.assertRaises(ValueError):preprocess_face(np.zeros((3,3,3),np.float32),(2,2))
        with self.assertRaises(ValueError):preprocess_face(np.zeros((3,3,4),np.uint8),(2,2))


if __name__=='__main__':unittest.main()
