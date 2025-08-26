# Tumor-analysis
This project builds a Tumor Image Classification system using Graph Neural Networks (GNNs) with Superpixels. Images are segmented into superpixels, converted into graph structures, and classified as tumor or normal. The GNN learns node features and adjacency relations, achieving efficient tumor detection with accuracy evaluation for medical image analysis and decision support.
1. Library Setup

Installs PyTorch, PyTorch Geometric (PyG), OpenCV, scikit-image, and other helper libraries.

These are needed to handle deep learning and graph-based operations.

🔹 2. Dataset Handling

Takes your archive (4).zip dataset.

You can either upload manually or load from Google Drive.

Extracts it into /content/tumor_dataset.

Inside, the code expects folders like tumor/ and normal/.

🔹 3. Image → Graph Conversion (Superpixels)

Each image is resized to 128x128.

Uses SLIC (Simple Linear Iterative Clustering) to break the image into ~75 superpixels.

Each superpixel becomes a node in a graph.

Node features = average LAB color of that region.

Edges = adjacency (neighboring superpixels are connected).

Finally, we get a graph object (torch_geometric.data.Data) with:

x → node features

edge_index → graph connections

y → label (0 = normal, 1 = tumor)

🔹 4. Train/Test Split

Splits graphs into 80% training and 20% testing.

Loads them into batches using PyG’s DataLoader.

🔹 5. GNN Model (TumorGNN)

Uses Graph Convolutional Network (GCNConv) layers.

Two convolution layers extract graph features.

Global mean pooling summarizes node info into a graph-level feature.

A final linear layer predicts whether the graph (image) is tumor or normal.

🔹 6. Training & Testing

Optimizer = Adam with learning rate 0.005.

Loss = CrossEntropyLoss (since it’s classification).

Runs for 10 epochs:

Each epoch trains the model on training graphs.

Calculates accuracy on both train and test sets.

🔹 7. Output

Prints loss + accuracy per epoch.

At the end, shows the Final Test Accuracy.
