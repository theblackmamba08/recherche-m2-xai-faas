# BRIEF — Architecture LSTM : du Neurone Biologique à la Mémoire Longue Terme

> Rempli a posteriori (présentation déjà réalisée).

- **Date** : 2025-11-25 (samedi — ~2 semaines après le 11/11/2025)
- **Audience** : Dr LACMOU ZEUTOUO Jerry, Dr DJOUMESSI Kerol, Pr KENGNE TCHENDJI Vianney, M. Yvan
- **Durée** : non précisée
- **Format** : oral / slides (PDF non disponible)

## Objectif

Démontrer la compréhension de l'architecture LSTM en retraçant la généalogie complète des modèles : neurone biologique → perceptron → MLP → RNN → LSTM. Montrer à chaque étape les limites qui ont motivé le passage à l'architecture suivante.

## Points clés présentés

1. **Neurone biologique** : structure, fonctionnement, signal électrique — point de départ conceptuel.
2. **Perceptron** : première modélisation mathématique du neurone, classification linéaire, limite : non-linéairement séparable.
3. **MLP (Multi-Layer Perceptron)** : ajout de couches cachées + rétropropagation, limite : ne gère pas les séquences temporelles.
4. **RNN (Recurrent Neural Network)** : introduction de la récurrence pour traiter les séquences, limite : gradient qui disparaît (*vanishing gradient*) sur les longues séquences.
5. **LSTM (Long Short-Term Memory)** : gates (input, forget, output) + cellule de mémoire — résout le vanishing gradient et gère les dépendances à long terme.

## Fil conducteur

À chaque étape : présentation de l'architecture → ses capacités → ses limites → pourquoi aller vers la suivante. Approche pédagogique par escalier de complexité croissante.

## Demandes à l'audience

Validation de la compréhension d'ensemble de l'architecture LSTM avant de passer aux Transformers.
