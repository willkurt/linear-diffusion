import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class LinearDiffusion:

    def __init__(self, image_size=28, latent_size=12):

        # assuming square images and latents/embeddings
        self.image_size = image_size
        self.latent_size = latent_size
        self._fit = False

    def fit(self, X, y):
        """
        X is the text labels
        y is the images
        a bit confusing but... it make the api usage consistent
        """
        self._build_image_encoder()
        self._build_text_encoder()
        # Get the image embeddings

        images_flat = y.reshape(y.shape[0], self.image_size * self.image_size)
        images_encoded = self.image_encoder.fit_transform(images_flat)

        # Get the label embeddings
        labels = np.array(X).reshape(-1, 1)
        label_embeddings = self.text_encoder.fit_transform(labels)

        # put it all together
        X_train, y_noise = self._create_features_noise(label_embeddings, images_encoded)
        self.model = LinearRegression().fit(X=X_train, y=y_noise)

        self._fit = True
        # I would like to be able to access the score method later, which might
        # mean refactoring this a bit more.
        return self

    def predict(self, X, seed=1337):
        assert(self._fit, "Please fit the model before running predict")
        labels = np.array(X).reshape(-1, 1)
        label_embeddings = self.text_encoder.transform(labels)
        X_test, noise_test = self._create_features_noise(label_embeddings, seed=seed)
        est_noise = self.model.predict(X_test)
        denoised = noise_test - est_noise
        est_imgs = self.image_encoder.inverse_transform(denoised).reshape(
            label_embeddings.shape[0], self.image_size, self.image_size)
        return est_imgs

    def _build_image_encoder(self):
        self.image_encoder = Pipeline([('scaler',
                                        StandardScaler()),
                                       ('PCA',
                                        PCA(n_components=self.latent_size * self.latent_size))])

    def _build_text_encoder(self):
        self.text_encoder = OneHotEncoder(categories='auto',
                                          drop='first',
                                          # should be 'sparse_output' when upgraded
                                          sparse=False,
                                          handle_unknown='error')

    def _create_interaction_terms(self, text_embeddings, image_embeddings):
        interactions = []
        for i in range(text_embeddings.shape[1]):
            interactions.append((image_embeddings * np.array([text_embeddings[:, i]]).T))
        return np.concatenate(interactions, axis=1)

    def _create_features_noise(self, text_embeddings, image_embeddings=None, std=1.0, seed=1337):
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0,
                           scale=std,
                           size=(text_embeddings.shape[0], self.latent_size * self.latent_size))
        if image_embeddings is not None:
            noised_embeddings = image_embeddings + noise
        else:
            noised_embeddings = noise
        interaction_terms = self._create_interaction_terms(text_embeddings, noised_embeddings)
        # we need the noise back directly since it will be used as a target in training
        features = np.concatenate([noised_embeddings,
                                   text_embeddings,
                                   interaction_terms], axis=1)
        return features, noise