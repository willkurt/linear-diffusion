import unittest
import mnist
import numpy as np
from lineardiffusion import LinearDiffusion


class LinearDiffusionTestCase(unittest.TestCase):

    def setUp(self):
        train_imgs = mnist.train_images()
        test_imgs = mnist.test_images()
        self.all_labels = [str(val) for val in np.concatenate([mnist.train_labels(), mnist.test_labels()])]
        self.all_imgs = np.concatenate([train_imgs, test_imgs])

    def test_init(self):
        ld = LinearDiffusion()
        # just sanity check initialization
        self.assertEqual(ld.image_size, 28.0)
        self.assertEqual(ld.latent_size, 12.0)
        self.assertFalse(ld._fit)

    def test_fit(self):
        ld = LinearDiffusion()
        self.assertFalse(ld._fit)
        ld.fit(self.all_labels, self.all_imgs)
        self.assertTrue(ld._fit)

    def test_predict(self):
        ld = LinearDiffusion()
        ld.fit(self.all_labels, self.all_imgs)
        sample_a = ['1', '2', '3']
        result_a = ld.predict(sample_a)
        self.assertEqual(result_a.shape[0], 3)
        # this is an invariant implied by the current way the model is built
        self.assertEqual(result_a.shape[1], result_a.shape[2])


if __name__ == '__main__':
    unittest.main()
