import CreateML
import Foundation
//1 MLDataTable is a brand new object used to create a table determined to train or evaluate a ML model. We split our data into trainingData and testingData. Like before, the ratio is 80-20 and the seed is 5. The seed refers to where the classifier should start from. Then we define a MLTextClassifier called spamClassifier with our training data, defining what values of the data are text and what values are labels.
let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/Document/MyML/spam.json"))
let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)
let spamClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "label")
//2 determine how accurate our classifier is. In the side pane, you’ll be able to see the percentage.
let trainingAccuracy = (1.0 - spamClassifier.trainingMetrics.classificationError) * 100
let validationAccuracy = (1.0 - spamClassifier.validationMetrics.classificationError) * 100
//3 check how the evaluation performed. (Remember that the evaluation is the results used on text which the classifier has not seen before and how accurate it got them.)
let evaluationMetrics = spamClassifier.evaluation(on: testingData)
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100
//4 create some metadata for the ML model like the author, description, and version. We use the write() function to save the model to the location
let metadata = MLModelMetadata(author: "ToanCT", shortDescription: "A model trained to classify spam messages", version: "1.0")
try spamClassifier.write(to: URL(fileURLWithPath: "/Users/Document/MyML/Save/SpamDetector.mlmodel"), metadata: metadata)
