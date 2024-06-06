import torch
import numpy as np
from sklearn.cluster import DBSCAN, AffinityPropagation
from lf2cs.tool.util_tools import Tools
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.decomposition import PCA


class ProduceClassExtended(object):
    def __init__(self, config, samples):
        super().__init__()
        self.config = config
        self.samples = samples
        self.classes = None
        self.lf2cs_out_dim = None
        self.class_per_num = None
        # self.transform = transform
        
        ## by default let the minimum number of samples in 1 cluster be 5% of the total train dataset size
        # if min_samples == None:
        #     min_samples = int(len(samples)*0.05) 
            
    #     self.class_per_num = self.n_sample // self.out_dim * ratio
        self.count = 0
        self.count_2 = 0
        self.class_num = None 
        
        # self.clustering = DBSCAN(eps=ep, min_samples=1).fit(samples)
        # self.classes = self.clustering.predict(samples)
        pass

    def init(self):

        features = self.extract_features( self.samples)
        # min_samples = int(0.001*len(self.samples))
        # min_samples = 10
        # # self.classes = self.dbscan_clustering(features, eps=self.eps, min_samples=min_samples)
        self.classes, self.lf2cs_out_dim = self.affinity_clustering(features)
        self.class_num = np.zeros(shape=(self.lf2cs_out_dim,), dtype=np.int32)
        self.class_per_num =  len(self.samples) // self.lf2cs_out_dim
        # class_per_num = self.n_sample // self.out_dim
        # self.class_num += class_per_num
        # for i in range(self.out_dim):
        #     self.classes[i * class_per_num: (i + 1) * class_per_num] = i
        #     pass
        # np.random.shuffle(self.classes)
        # pass

    def extract_features(self, image_paths):
        vgg19 = models.vgg19(weights='VGG19_Weights.DEFAULT').features.eval().cuda()
        # resnet = resnet.eval().cuda()
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        
    
        features = []
        for _, _ , image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            img = self.config.transform_train_fsl(img)
            img = transforms.Resize((224, 224))(img)
            img = img.unsqueeze(0).cuda()
            with torch.no_grad():
                feature = vgg19(img).squeeze().cpu().numpy().flatten()
            features.append(feature)
        return np.array(features)

    def dbscan_clustering(self, features, eps, min_samples=20):
        print(features.shape, min_samples, eps)
        # scaler = StandardScaler()
        # features_scaled = scaler.fit_transform(features)
    
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs = self.config.num_workers)
        labels = dbscan.fit_predict(features)
          # Print clustering results
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            Tools.print(f"Cluster {label}: {len(cluster_indices)} images", txt_path=self.config.log_file)

        return labels, len

    def affinity_clustering(self, features):
        # print(features.shape, min_samples, eps)
        # scaler = StandardScaler()
        # features_scaled = scaler.fit_transform(features)
        # define the model
        model = AffinityPropagation(damping=0.9)
        
        # train the model
        # model.fit(features)
        model.fit(features)
        
        # assign each data point to a cluster
        labels = model.predict(features)
        
        # get all of the unique clusters
        unique_labels = np.unique(labels)

    
        # dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs = self.config.num_workers)
        # labels = dbscan.fit_predict(features)
        #   # Print clustering results
        # unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            Tools.print(f"Cluster {label}: {len(cluster_indices)} images", txt_path=self.config.log_file)

        return labels, len(unique_labels)



    def reset(self):
        self.count = 0
        self.count_2 = 0
        self.class_num *= 0
        pass

    def cal_label(self, out, indexes):
        top_k = out.data.topk(self.lf2cs_out_dim, dim=1)[1].cpu()
        indexes_cpu = indexes.cpu()

        batch_size = top_k.size(0)
        class_labels = np.zeros(shape=(batch_size,), dtype=np.int32)

        for i in range(batch_size):
            for j_index, j in enumerate(top_k[i]):
                #if self.class_per_num > self.class_num[j]:
                class_labels[i] = j
                self.class_num[j] += 1
                self.count += 1 if self.classes[indexes_cpu[i]] != j else 0
                self.classes[indexes_cpu[i]] = j
                self.count_2 += 1 if j_index != 0 else 0
                break
                # pass
            pass
        pass

    def get_label(self, indexes):
        _device = indexes.device
        return torch.tensor(self.classes[indexes.cpu()]).long().to(_device)

    # pass

# Function to extract features from images using a pre-trained ResNet model
# Function to perform DBSCAN clustering on image features
# def dbscan_clustering(features, eps=0.5, min_samples=5):
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)

#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(features_scaled)
#     return labels

# # Example usage
# if __name__ == "__main__":
#     # Directory containing images
#     image_dir = "images/"
#     image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

#     # Extract features from images
#     features = extract_features(image_paths)

#     # Perform DBSCAN clustering
#     eps = 0.5  # Neighborhood radius
#     min_samples = 5  # Minimum number of points in a neighborhood
#     labels = dbscan_clustering(features, eps=eps, min_samples=min_samples)

  
