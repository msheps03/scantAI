# scantAI
Evaluating Synthetic Datatset Generation Techniques on One Class Classification of Ticks

Final code is available in "ConflictOfInterest Copy.ipynb"

*Image files for this project cannot be released except for academic purposes*

In conjuction with 3D scanner, scAnt. The forked repository is available here: [Shepherd_ScAnt](https://github.com/msheps03/scant_shepherd)
and original here: [scAnt](https://github.com/evo-biomech/scAnt)

Abstract: Image classification has long been an AI challenge. Gathering datasets to train AI models is a particularly troublesome issue. Creating these datasets synthetically reduces the cost of generating these datasets however, the models trained with these datasets tend to suffer from poor accuracy. By using feature extraction performed on a ResNet50 Convolution Neural Network (CNN). These features will be used to train a One Class Classifier (OCC). Gaussian Mixtures are used to classify the specimen, in this study that is Ticks. Ticks are parasitic disease carriers and being able to determine quickly and confidently if that bug is a tick helps one make an informed decision about the appropriate action. Ticks are small and intricate making them good specimens to analyze for this study. Demonstrating effective classification on ticks with a synthetically trained OCC would show that synthetic generation techniques are comparable to using real images. To generate these datasets 2D augmentation and 3D scenes will be used. 2D augmentation modifies an existing image to generate multiple similar images to expand the dataset. 3D scenes will be rendered in Blender and use a 3D model to synthetically generate images with varying perspectives and shadows. These generation techniques will be evaluated by accuracy and their “cost to generate”.

Results: Utilizing synthetic datasets to train classifiers is possible, however utilizing 3D models and by extension 3D scenes to train a dataset is too computationally expensive. There has been discussion of utilizing AI models to generate 3D scenes in Blender, as this technology becomes more avaiable I would confidently say that datasets generated with 3D scenes would be effective assuming the 3D model is comprehensive of the test set. In this instance unfortunately only one 3D model was used, and by only analyzing the on species of tick the model is likely to misclassify any of the numerous other species. In this study specifically using the three genera of tick species would solve the issue of a poorly representative test set.
