package adampiech.convnet.controll.engine;

import adampiech.convnet.controll.engine.convNETLayers.ConvNETLayer;
import adampiech.convnet.controll.engine.convNETLayers.PoolingLayer;
import adampiech.convnet.controll.engine.convNETLayers.artificialNeuralNetwork.ANN;
import adampiech.convnet.controll.engine.convNETLayers.artificialNeuralNetwork.Perceptron;
import adampiech.convnet.controll.services.imageServices.ImageServices;
import adampiech.convnet.controll.services.matrixServices.MatrixServices;
import adampiech.convnet.controll.services.neuralNetworkServices.AdapterNN;
import adampiech.convnet.controll.utils.TrainingData;
import org.json.JSONArray;
import org.json.JSONObject;
import org.la4j.Matrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Timer;

import static adampiech.convnet.controll.services.matrixServices.MatrixServices.copyMatrix;

/**
 * Created by Adam Piech on 2016-12-28.
 */
public class Training {

    private AdapterNN adapter = new AdapterNN();

    public void train(List<TrainingData> trainingData, List<ConvNETLayer> convNETLayers, PoolingLayer poolingLayer, ANN ann) {

        int counter = 0;

        for (TrainingData data : trainingData) {

            System.out.println(getClass().getName() + " --> " + "PROCESSING " + ++counter + " FROM " + trainingData.size() + " LAYERS.");
            long start = System.currentTimeMillis();

            Matrix[] processData = copyMatrix(data.getImage());
            for (ConvNETLayer cnnLayer : convNETLayers) {
                processData = poolingLayer.processLayer(cnnLayer.processLayer(processData, null));
//                        m -> ImageServices.saveImage(MatrixServices.matrixToMat(m), "working")));
            }

            double[] deltasANN =  ann.train(adapter.cnnToANN(processData), data.getTarget());
            Matrix[] deltasCNN = adapter.annToCNN(deltasANN);

            for (int index = convNETLayers.size() - 1; index >= 0; index--) {
                deltasCNN = convNETLayers.get(index).backPropagation(deltasCNN);
            }

            System.out.println(getClass().getName() + " --> " + "LAYER PROCESSING TIME: " + (System.currentTimeMillis() - start) / 1000 + "s.");
        }
    }

    public double[] test(Matrix[] data, List<ConvNETLayer> convNETLayers, PoolingLayer poolingLayer, ANN ann) {
        for (ConvNETLayer cnnLayer : convNETLayers) {
            data = poolingLayer.processLayer(cnnLayer.processLayer(data, null));
        }

        double[] result;
        System.out.print(getClass().getName() + " --> " + "RESULT: ");
        Arrays.stream(ann.process(result = adapter.cnnToANN(data))).forEach(value -> System.out.print(value + ", "));
        System.out.println();

        return result;
    }

//    private void saveNetworkToJsonFile(List<ConvNETLayer> convNETLayers, int[] annArchitecture, ANN ann) {
//        JSONArray jsonCNNArray = createJsonCNNObject(convNETLayers);
//        JSONObject jsonANN = createJsonANNObject(ann, annArchitecture);
//        JSONObject jsonNetwork = new JSONObject();
//        jsonNetwork.put("ConvNET", jsonCNNArray);
//        jsonNetwork.put("ANN", jsonANN);
//
//        try {
//            FileWriter file = new FileWriter("test.json");
//            file.write(jsonNetwork.toString(3).replace("\\", "\n"));
//            file.flush();
//            file.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }

//    private JSONArray createJsonCNNObject(List<ConvNETLayer> convNETLayers) {
//        JSONArray jsonCNNArray = new JSONArray();
//        for (ConvNETLayer cnnLayer : convNETLayers) {
//
//            JSONObject jsonCNNObject = new JSONObject();
//
//            JSONArray jsonWeightArray = new JSONArray();
//            for (Matrix[] weightSubLayer : cnnLayer.getWeights()) {
//                JSONArray jsonWeightDepthArray = new JSONArray();
//                for (Matrix weightDepth : weightSubLayer) {
//                    JSONArray jsonWeightColArray = new JSONArray();
//                    for (int col = 0; col < weightDepth.columns(); col++) {
//                        JSONArray jsonWeightRowArray = new JSONArray();
//                        for (int row = 0; row < weightDepth.rows(); row++) {
//                            jsonWeightRowArray.put(weightDepth.get(row, col));
//                        }
//                        jsonWeightColArray.put(jsonWeightRowArray);
//                    }
//                    jsonWeightDepthArray.put(jsonWeightColArray);
//                }
//                jsonWeightArray.put(jsonWeightDepthArray);
//            }
//            jsonCNNObject.put("weights", jsonWeightArray);
//
//            JSONArray jsonBiasArray = new JSONArray();
//            for (double bias : cnnLayer.getBiases()) {
//                jsonBiasArray.put(bias);
//            }
//            jsonCNNObject.put("biases", jsonBiasArray);
//
//            jsonCNNObject.put("size", cnnLayer.getSize());
//            jsonCNNObject.put("length", cnnLayer.getLength());
//            jsonCNNObject.put("receptiveField", cnnLayer.getReceptiveField());
//            jsonCNNObject.put("depth", cnnLayer.getDepth());
//            jsonCNNObject.put("stride", cnnLayer.getStride());
//            jsonCNNObject.put("zeroPadding", cnnLayer.getZeroPadding());
//
//            jsonCNNArray.put(jsonCNNObject);
//        }
//        return jsonCNNArray;
//    }

//    private JSONObject createJsonANNObject(ANN ann, int[] annArchitecture) {
//        JSONObject jsonANNObject = new JSONObject();
//
//        JSONArray jsonANNArchitectureArray = new JSONArray();
//        for (int layerSize : annArchitecture) {
//            jsonANNArchitectureArray.put(layerSize);
//        }
//        jsonANNObject.put("ANNArchitecture", jsonANNObject);
//
//        JSONArray jsonLayerArray = new JSONArray();
//        for (ANN.Layer layer : ann.getLayers()) {
//            JSONArray jsonPerceptronsArray = new JSONArray();
//            for (Perceptron perceptron : layer.getPerceptrons()) {
//                JSONArray jsonWeightsArray = new JSONArray();
//                for (double weight : perceptron.getWeights()) {
//                    jsonWeightsArray.put(weight);
//                }
//                jsonPerceptronsArray.put(jsonWeightsArray);
//            }
//            jsonLayerArray.put(jsonPerceptronsArray);
//        }
//        jsonANNObject.put("Layers", jsonLayerArray);
//        return jsonANNObject;
//    }

}
