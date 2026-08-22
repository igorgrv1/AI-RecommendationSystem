# AI Recommendation System — TensorFlow.js Step-by-Step Guide

This project trains a lightweight neural network **directly in the browser** using [TensorFlow.js](https://www.tensorflow.org/js) and a Web Worker. The model learns to score how well a product matches a user based on their purchase history and profile.

![demo](demo.png)
---

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [How Data Flows Through the System](#2-how-data-flows-through-the-system)
3. [Step 1 — Build the Global Context](#3-step-1--build-the-global-context)
4. [Step 2 — Normalize Continuous Values](#4-step-2--normalize-continuous-values)
5. [Step 3 — Encode a Product into a Feature Vector](#5-step-3--encode-a-product-into-a-feature-vector)
6. [Step 4 — Encode a User into a Feature Vector](#6-step-4--encode-a-user-into-a-feature-vector)
7. [Step 5 — Build Training Data](#7-step-5--build-training-data)
8. [Step 6 — Define and Train the Neural Network](#8-step-6--define-and-train-the-neural-network)
9. [Step 7 — Run Inference (Get Recommendations)](#9-step-7--run-inference-get-recommendations)
10. [Feature Weights Reference](#10-feature-weights-reference)
11. [Normalized Values Reference](#11-normalized-values-reference)
12. [Architecture Summary](#12-architecture-summary)

---

## 1. Project Setup

Install dependencies and start the application:

```bash
npm install
npm start
```

Open your browser at `http://localhost:8080`. The UI lets you select a user, view their purchase history, and trigger model training. All training runs inside a Web Worker so it never blocks the main thread.

---

## 2. How Data Flows Through the System

```
products.json + users
        │
        ▼
  makeContext()          ← compute global min/max for price & age,
        │                  unique lists for colors & categories,
        │                  average buyer age per product
        ▼
  encodeProduct()        ← turn each product into a numeric vector
  encodeUser()           ← turn each user into a numeric vector
        │
        ▼
  createTrainingData()   ← pair every (user vector, product vector)
        │                  label = 1 if user bought that product, else 0
        ▼
  configureNeuralNetAndTrain()   ← build + train the model
        │
        ▼
  recommend()            ← encode target user, score all products,
                           return ranked list
```

---

## 3. Step 1 — Build the Global Context

**File:** [`makeContext()`](src/workers/modelTrainingWorker.js)

Before any encoding can happen, the system computes global statistics from the full dataset. These statistics are used to normalize every feature consistently across users and products.

What gets computed:

| Statistic | Description |
|---|---|
| `minAge` / `maxAge` | Minimum and maximum user ages across all users |
| `minPrice` / `maxPrice` | Minimum and maximum product prices across all products |
| `colors` | Deduplicated list of all product colors (used to build the one-hot index) |
| `categories` | Deduplicated list of all product categories (used to build the one-hot index) |
| `productAvgAgeNorm` | For each product, the normalized average age of all users who bought it |
| `dimentions` | Total vector length = `2 + numCategories + numColors` (price slot + age slot + one-hot slots) |

**Why a shared context?**  
All features must be encoded using the same scale. If training encodes price with a different min/max than inference, the model receives inconsistent inputs and produces wrong predictions.

---

## 4. Step 2 — Normalize Continuous Values

**File:** [`normalize()`](src/workers/modelTrainingWorker.js)

```js
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)
```

### What is normalization?

Normalization maps any number onto the `0–1` range. Without it, a feature like price (`$39.99–$199.99`) would have much larger raw values than age expressed as a fraction, causing the neural network to over-weight whichever feature has larger numbers — regardless of its actual importance.

### Formula

```
normalized = (value - min) / (max - min)
```

The `|| 1` guard prevents division by zero when all values in the dataset are identical.

### Worked Examples

#### Price normalization

Assume the product catalog has prices ranging from `$39.99` to `$199.99`.

| Raw price | Formula | Normalized |
|---|---|---|
| `$39.99` | `(39.99 - 39.99) / (199.99 - 39.99)` | `0.00` (cheapest) |
| `$99.99` | `(99.99 - 39.99) / (199.99 - 39.99)` | `0.375` |
| `$129.99` | `(129.99 - 39.99) / (199.99 - 39.99)` | `0.5625` |
| `$199.99` | `(199.99 - 39.99) / (199.99 - 39.99)` | `1.00` (most expensive) |

#### Age normalization

Assume users range from age `18` to `60`.

| Raw age | Formula | Normalized |
|---|---|---|
| `18` | `(18 - 18) / (60 - 18)` | `0.00` (youngest) |
| `27` | `(27 - 18) / (60 - 18)` | `0.214` |
| `39` | `(39 - 18) / (60 - 18)` | `0.5` |
| `60` | `(60 - 18) / (60 - 18)` | `1.00` (oldest) |

#### Product average buyer age normalization

For each product, the system computes the mean age of all users who purchased it, then normalizes that average with the same `minAge`/`maxAge` formula. If no one has bought the product yet, the midpoint `(minAge + maxAge) / 2` is used as a fallback, which normalizes to `0.5`.

This gives the model a signal for "what age group tends to buy this product" without leaking raw ages.

---

## 5. Step 3 — Encode a Product into a Feature Vector

**File:** [`encodeProduct()`](src/workers/modelTrainingWorker.js)

Every product is turned into a flat numeric array (a TensorFlow `Tensor1D`) before it can be used in training or inference. The vector has this layout:

```
[ price_weighted, avgBuyerAge_weighted, ...categoryOneHot_weighted, ...colorOneHot_weighted ]
```

### Continuous features

Both continuous features are normalized to `0–1` first and then multiplied by their weight:

```js
// Price slot
normalizedPrice * WEIGHTS.price      // e.g. 0.5625 * 0.2 = 0.1125

// Average buyer age slot
productAvgAgeNorm[product.name] * WEIGHTS.age   // e.g. 0.214 * 0.1 = 0.0214
```

### Categorical features — one-hot encoding

Categories and colors cannot be passed to a neural network as raw strings. **One-hot encoding** converts them to a binary array where exactly one position is `1` (the matching category/color) and all others are `0`.

**Example — categories = `['accessories', 'electronics', 'clothing']`**

| Product category | One-hot vector |
|---|---|
| `accessories` | `[1, 0, 0]` |
| `electronics` | `[0, 1, 0]` |
| `clothing` | `[0, 0, 1]` |

After encoding, each `1` is multiplied by the category weight (`0.4`), so the active slot becomes `0.4` instead of `1`.

**Example — colors = `['black', 'grey', 'blue']`**

| Product color | One-hot vector | After weight (×0.3) |
|---|---|---|
| `black` | `[1, 0, 0]` | `[0.3, 0, 0]` |
| `grey` | `[0, 1, 0]` | `[0, 0.3, 0]` |
| `blue` | `[0, 0, 1]` | `[0, 0, 0.3]` |

### Complete product vector example

Given: price `$129.99`, avgBuyerAge normalized `0.214`, category `accessories` (index 0 of 3), color `black` (index 0 of 3):

```
[
  0.1125,        // normalized price × 0.2
  0.0214,        // avg buyer age normalized × 0.1
  0.4, 0.0, 0.0, // one-hot category × 0.4  (accessories active)
  0.3, 0.0, 0.0  // one-hot color × 0.3     (black active)
]
```

---

## 6. Step 4 — Encode a User into a Feature Vector

**File:** [`encodeUser()`](src/workers/modelTrainingWorker.js)

A user vector represents the user's "taste profile" in the same numeric space as the product vectors. This makes it possible to concatenate the two and feed them to the model as a single input.

### User with purchases

Each purchased product is encoded using `encodeProduct()`. All resulting vectors are stacked and averaged element-by-element (`tf.stack(...).mean(0)`). The result is a single vector in the same shape as a product vector, representing the centroid of everything the user has bought.

```
userVector = mean( encodeProduct(purchase1), encodeProduct(purchase2), ... )
```

### User with no purchases (cold start)

When a user has no purchase history the system falls back to a sparse vector:
- Price slot: `0` (no price signal)
- Age slot: `normalizedAge * WEIGHTS.age` (only the user's own age is used)
- Category slots: all `0` (no category signal)
- Color slots: all `0` (no color signal)

This fallback means new users receive recommendations that are closer to globally-popular products rather than personalized ones — a reasonable cold-start behaviour.

---

## 7. Step 5 — Build Training Data

**File:** [`createTrainingData()`](src/workers/modelTrainingWorker.js)

Training examples are constructed by pairing every user vector with every product vector:

```
input  = [ ...userVector, ...productVector ]   // concatenated
label  = 1 if the user purchased that product, else 0
```

Because this produces one row per `(user × product)` combination, only users who have at least one purchase are included — users with no history provide no positive labels and would only add noise.

The final tensors:

| Tensor | Shape | Description |
|---|---|---|
| `xs` | `[numExamples, dimentions × 2]` | Input features |
| `ys` | `[numExamples, 1]` | Binary labels (bought / not bought) |

---

## 8. Step 6 — Define and Train the Neural Network

**File:** [`configureNeuralNetAndTrain()`](src/workers/modelTrainingWorker.js)

### Architecture

The model is a simple feedforward network with three hidden layers followed by a sigmoid output:

```
Input layer   →  Dense(128, relu)
Hidden layer 1 →  Dense(64, relu)
Hidden layer 2 →  Dense(32, relu)
Output layer  →  Dense(1, sigmoid)
```

| Layer | Units | Activation | Purpose |
|---|---|---|---|
| Input | 128 | ReLU | Broad pattern detection across all features |
| Hidden 1 | 64 | ReLU | Start compressing into higher-level combinations |
| Hidden 2 | 32 | ReLU | Distill the most relevant signals |
| Output | 1 | Sigmoid | Produce a score in [0, 1] |

**Why ReLU?**  
ReLU (`max(0, x)`) discards negative activations, helping the network learn non-linear decision boundaries without the vanishing-gradient problems of sigmoid/tanh in hidden layers.

**Why Sigmoid on the output?**  
Sigmoid maps any real number to `[0, 1]`, turning the final neuron into a probability-like score. A score near `1.0` means the model strongly predicts the user would buy that product; near `0.0` means the opposite.

### Compilation

```js
model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
})
```

| Parameter | Value | Reason |
|---|---|---|
| Optimizer | Adam (lr = 0.01) | Adaptive learning rate, converges well on small datasets |
| Loss | Binary cross-entropy | Standard loss for binary classification (bought / not bought) |
| Metric | Accuracy | Human-readable progress indicator during training |

### Training hyperparameters

```js
model.fit(xs, ys, {
    epochs: 100,
    batchSize: 32,
    shuffle: true
})
```

| Parameter | Value | Meaning |
|---|---|---|
| `epochs` | 100 | Number of full passes over the training set |
| `batchSize` | 32 | Number of examples processed per gradient update |
| `shuffle` | true | Randomise example order each epoch to reduce overfitting |

Progress (epoch number, loss, accuracy) is streamed back to the UI via `postMessage` after every epoch.

---

## 9. Step 7 — Run Inference (Get Recommendations)

**File:** [`recommend()`](src/workers/modelTrainingWorker.js)

Once training is complete, recommendations for a given user are produced in four steps:

**Step 1 — Encode the user**

```js
const userVector = encodeUser(user, context).dataSync()
```

The target user is encoded using the same `encodeUser()` function used during training, producing a flat numeric array.

**Step 2 — Build input pairs**

```js
const inputs = context.productVectors.map(({ vector }) => [...userVector, ...vector])
```

The user vector is concatenated with every stored product vector, forming one input row per product.

**Step 3 — Predict scores in a single batch**

```js
const inputTensor = tf.tensor2d(inputs)
const predictions = _model.predict(inputTensor)
const scores = predictions.dataSync()
```

All `(user, product)` pairs are fed to the model in one batched call. Each prediction is a number between `0` and `1`.

**Step 4 — Sort and return**

```js
recommendations.sort((a, b) => b.score - a.score)
```

Products are ranked by descending score and posted back to the UI thread.

> **Production tip:** For large catalogs, store product vectors in a vector database (e.g., Postgres with `pgvector`, Pinecone, or Neo4j). At inference time, retrieve only the top-K nearest neighbours to the user vector, then run `_model.predict()` on that smaller candidate set instead of all products.

---

## 10. Feature Weights Reference

Weights are applied after normalization and one-hot encoding to control the relative importance of each feature group during training:

```js
const WEIGHTS = {
    category: 0.4,
    color:    0.3,
    price:    0.2,
    age:      0.1,
}
```

| Feature | Weight | Interpretation |
|---|---|---|
| `category` | `0.4` | Strongest signal — what type of product it is matters most |
| `color` | `0.3` | Second strongest — color preference is a meaningful personal signal |
| `price` | `0.2` | Moderate — price range gives budget signal |
| `age` | `0.1` | Weakest — average buyer age is a soft demographic hint |

Weights must not necessarily sum to `1.0`. They scale the magnitude of each feature group in the final vector.

---

## 11. Normalized Values Reference

A summary of every value that gets normalized, where the bounds come from, and how the formula is applied:

| Value | Min source | Max source | Formula | Fallback |
|---|---|---|---|---|
| **Product price** | `Math.min(...allProductPrices)` | `Math.max(...allProductPrices)` | `(price - minPrice) / (maxPrice - minPrice)` | — |
| **User age** | `Math.min(...allUserAges)` | `Math.max(...allUserAges)` | `(age - minAge) / (maxAge - minAge)` | — |
| **Product avg buyer age** | Same `minAge` | Same `maxAge` | `(avgAge - minAge) / (maxAge - minAge)` | `0.5` (midpoint) if no buyers yet |
| **Category (one-hot)** | Index `0` | `numCategories - 1` | Position `categoriesIndex[category]` set to `weight`, rest `0` | — |
| **Color (one-hot)** | Index `0` | `numColors - 1` | Position `colorsIndex[color]` set to `weight`, rest `0` | — |

All normalized values are then **multiplied by their respective weight** before being concatenated into the final vector. This means the actual range in the vector is `[0, weight]` rather than `[0, 1]`.

---

## 12. Architecture Summary

```
products.json ──► makeContext() ──► encode*() ──► createTrainingData()
                                                         │
                                                         ▼
                                             configureNeuralNetAndTrain()
                                               Dense(128) → Dense(64)
                                               → Dense(32) → Dense(1, sigmoid)
                                               optimizer: Adam(0.01)
                                               loss: binaryCrossentropy
                                               epochs: 100 | batchSize: 32
                                                         │
                                                         ▼
                                                  recommend()
                                               score all products → sort → UI
```

### Project Structure

```
src/
├── workers/
│   └── modelTrainingWorker.js   ← TensorFlow.js training & inference (Web Worker)
├── events/
│   └── constants.js             ← Worker message event names
├── view/                        ← DOM management and templates
├── controller/                  ← Connects views and services
├── service/                     ← Business logic
└── data/
    └── products.json            ← Product catalog
index.html
index.js
```
