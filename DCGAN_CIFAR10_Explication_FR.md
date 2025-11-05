# 🎨 DCGAN — Réseau Génératif Adversarial Convolutif sur CIFAR-10
*Documentation technique et guide d’utilisation – en français*

## 1. Introduction
Les **GANs** (*Generative Adversarial Networks*) permettent de créer de nouvelles images à partir d’un bruit aléatoire.  
Un **DCGAN** (*Deep Convolutional GAN*) est une version améliorée adaptée aux images en couleur grâce à des couches de convolution et de déconvolution.  

### 🎯 Objectif du projet
Générer des images **RVB (32×32)** inspirées du jeu de données **CIFAR-10** à l’aide d’un générateur et d’un discriminateur entraînés en compétition.  

---

## 2. Structure du notebook
Le notebook se compose de plusieurs sections :

| Section | Description |
|----------|--------------|
| Importations | Chargement de PyTorch, torchvision et utilitaires. |
| Configuration | Définition des hyperparamètres (taille du lot, taux d’apprentissage, nombre d’époques, etc.). |
| Dataset | Chargement de CIFAR-10 et normalisation dans l’intervalle [-1, 1]. |
| Modèles | Définition des classes `Generator` et `Discriminator`. |
| Boucle d’entraînement | Mise à jour alternée des réseaux G et D. |
| Visualisation | Sauvegarde et affichage des échantillons à chaque époque. |
| Sauvegarde des modèles | Enregistrement des poids au format `.pth`. |

---

## 3. Explication technique détaillée  

### 🧩 Architecture du Générateur  
Le **générateur (G)** reçoit un vecteur de bruit `z` (100 valeurs aléatoires) et le transforme en image RGB (3 canaux, 64×64) par une succession de **couches de convolution transposée** :

```
z (100,1,1)
 ├─ ConvTranspose2d → 4x4x512
 ├─ BatchNorm2d + ReLU
 ├─ ConvTranspose2d → 8x8x256
 ├─ ConvTranspose2d → 16x16x128
 ├─ ConvTranspose2d → 32x32x64
 ├─ ConvTranspose2d → 64x64x3
 └─ Tanh → image finale [-1,1]
```

Chaque couche agrandit spatialement l’image et affine les détails.  
La fonction **Tanh** assure que la sortie soit normalisée dans \([-1,1]\), correspondant à la normalisation du dataset.  

### 🔍 Architecture du Discriminateur  
Le **discriminateur (D)** reçoit une image (réelle ou générée) et apprend à distinguer les vraies des fausses :

```
Image (3,64,64)
 ├─ Conv2d → 32x32x64
 ├─ LeakyReLU(0.2)
 ├─ Conv2d → 16x16x128
 ├─ BatchNorm2d + LeakyReLU
 ├─ Conv2d → 8x8x256
 ├─ Conv2d → 4x4x512
 ├─ Conv2d → 1x1x1
 └─ Sigmoid → probabilité "réelle"
```

Le **LeakyReLU** évite le problème du *dead ReLU*, et la **Sigmoid** produit une probabilité entre 0 et 1.  

### ⚖️ Fonction de perte  
Le DCGAN optimise un **jeu min-max** entre les deux réseaux :

\[
\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]
\]

- Le **discriminateur** maximise la probabilité d’identifier correctement les vraies images.  
- Le **générateur** cherche à minimiser la capacité de D à les distinguer.  

### 🔧 Optimisateurs  
Utilisation d’**Adam** avec `lr=0.0002`, `betas=(0.5, 0.999)` pour une convergence stable.

---

## 4. Guide d’utilisation (Google Colab)  

### ⚙️ Étapes principales  
1. **Importer le notebook `.ipynb` dans Google Colab**.  
2. Aller dans le menu **Exécution → Modifier le type d’exécution → GPU**.  
3. Lancer les cellules dans l’ordre :  
   - Installation et importations  
   - Chargement du dataset  
   - Définition du modèle  
   - Entraînement  
4. À chaque époque, une image `epoch_XXX.png` s’affiche dans `/content/samples_color_gan/`.  

### 💾 Sauvegarde des modèles  
Les poids sont enregistrés sous :  
```
checkpoints/G_epoch_XXX.pth
checkpoints/D_epoch_XXX.pth
```

Ces fichiers peuvent être rechargés pour générer de nouvelles images sans ré-entraîner le réseau.  

---

## 5. Générer de nouvelles images avec un modèle sauvegardé  

```python
import torch
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt
from dcgan_color import Generator  # importer la même classe que celle de l’entraînement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(z_dim=100, channels=3)
G.load_state_dict(torch.load("checkpoints/G_epoch_025.pth", map_location=device))
G.eval()

z = torch.randn(64, 100, 1, 1, device=device)
with torch.no_grad():
    fake = G(z).cpu()

grid = make_grid((fake + 1) / 2, nrow=8)
save_image(grid, "images_generees.png")

plt.figure(figsize=(6,6))
plt.axis("off")
plt.imshow(grid.permute(1,2,0))
plt.show()
```

---

## 6. Interprétation des résultats  
- **Loss_D** : diminue quand le discriminateur apprend à détecter les fausses images.  
- **Loss_G** : diminue quand le générateur produit des images plus réalistes.  
- Les premiers échantillons sont flous ; après une dizaine d’époques, on observe des contours et des couleurs plausibles.  

### 🎞️ Visualisation complète  
Un GIF d’évolution peut être généré avec :

```python
import imageio, glob
frames = [imageio.imread(p) for p in sorted(glob.glob(f"{out_dir}/epoch_*.png"))]
imageio.mimsave(f"{out_dir}/evolution_cifar10.gif", frames, fps=2)
```

---

## 7. Améliorations possibles  
- **WGAN-GP** pour une stabilité renforcée (perte de Wasserstein).  
- **Spectral Normalization** pour un contrôle des gradients.  
- **Augmentation** légère du dataset (flip, rotation).  
- **StyleGAN** pour un contrôle fin du style et de la variabilité.  
