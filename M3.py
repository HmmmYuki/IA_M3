from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

# 1. DONNÉES : X = les mesures, y = l'espèce à deviner
iris = load_iris()
X = iris.data        # 150 fleurs x 4 mesures (4 neurone en entrée car il y a 4 mesures)
y = iris.target      # 150 étiquettes (0, 1 ou 2) (3 neurone en sortie car on a 3 catégorie)

# 2. MODÈLE : un réseau avec UNE couche cachée de 10 neurones
reseau = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000)

# 3. ENTRAÎNEMENT : le réseau ajuste tous ses poids automatiquement
reseau.fit(X, y)

# 4. ÉVALUATION : quel pourcentage de fleurs bien classées ?
score = reseau.score(X, y)
print("Précision :", score)        # ex. : 0.98  ->  98 % de bonnes réponses !

# 5. PRÉDICTION sur une nouvelle fleur jamais vue
nouvelle_fleur = [[5.1, 3.5, 1.4, 0.2]]
print("Espèce prédite :", reseau.predict(nouvelle_fleur))