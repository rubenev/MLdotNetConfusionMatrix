using System;
using System.Linq;

using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Some.Actions
{
    public class CreateConfusionMatrix
    {
        private ITransformer loadedModel;
        private readonly FileSystemRepository fileSystemRepository;

        public CreateConfusionMatrix(FileSystemRepository fileSystemRepository)
        {
            this.fileSystemRepository = fileSystemRepository;
        }

        public CreateConfusionMatrix() : this(
        fileSystemRepository: new FileSystemRepository())
        {
        }

        private (int, string[], bool[]) PullDataFromDataView(IDataView testData, Guid modelId, MLContext mlContext, string featuresColumn, string labelsColumn)
        {          
            var sizeTestSet = testData.GetColumn<string>(mlContext, featuresColumn).Count(); 
            var testSetFeatures = testData.GetColumn<string>(mlContext, featuresColumn).Take(sizeTestSet).ToArray(); 
            var testSetLabels = testData.GetColumn<bool>(mlContext, labelsColumn).Take(sizeTestSet).ToArray(); 
            this.loadedModel = mlContext.Model.Load(this.fileSystemRepository.GetModelFileStream(modelId));

            return (sizeTestSet, testSetFeatures, testSetLabels);
        }

        public ConfusionMatrixModel ConstructConfusionMatrix(IDataView testData, Guid modelId, MLContext mlContext, string featuresColumn, string labelsColumn)
        {
            var (sizeTestSet, testSetFeatures, testSetLabels) = PullDataFromDataView(testData, modelId, mlContext, featuresColumn, labelsColumn);
            var confusionMatrixModel = new ConfusionMatrixModel();
            var predictionFunction = this.loadedModel.CreatePredictionEngine<ClassifierDataSet, ClassifierPrediction>(mlContext);
            for (var i = 0; i < sizeTestSet; i++)
            {
                var prediction = predictionFunction.Predict(new ClassifierDataSet()
                {
                    SentimentText = testSetFeatures[i]
                });
                if (prediction.Prediction)
                {
                    if (testSetLabels[i])
                    {
                        confusionMatrixModel.TruePositives++;
                    }
                    else
                    {
                        confusionMatrixModel.FalsePositives++;
                    }
                }
                else
                {
                    if (testSetLabels[i] == false)
                    {
                        confusionMatrixModel.TrueNegatives++;
                    }
                    else
                    {
                        confusionMatrixModel.FalseNegatives++;
                    }
                }
            }

            return confusionMatrixModel;
        }
    }
}
