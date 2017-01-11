package adampiech.convnet.controll.engine.convNETLayers.artificialNeuralNetwork;

import static java.lang.Math.*;

/**
 * Created by Adam Piech on 2016-11-07.
 */

public class ANN {

    private int[] networkArchitecture;
    private Layer[] layers;

    public ANN(int[] networkArchitecture) {
        this.networkArchitecture = networkArchitecture;
        layers = new Layer[networkArchitecture.length];
    }

    public double[] train(double[] input, double[] target) {
        double[] output = null;
        double[] result = null;
        if (input.length == networkArchitecture[0]) {
            do {
                output = process(input);
                result = backPropagation(target);
            } while (!outputIsSameAsTarget(output, target));
            return result;
        } else {
            System.out.println(getClass().getName() + " --> " + "LAYER SIZE MUST BE EQUAL: " + input.length);
        }
        return null;
    }

    public double[] process(double[] inputs) {
        double[] layerInOut = inputs;
        for (int layerLevel = 0; layerLevel < networkArchitecture.length; layerLevel++) {
            if (layers[layerLevel] == null) {
                layers[layerLevel] = new Layer(networkArchitecture[layerLevel], layerInOut.length);
            }
            layerInOut = layers[layerLevel].countOutputLayer(layerInOut);
        }
        return layerInOut;
    }

    public double[] backPropagation(double[] target) {
        double[] deltas = null;
        int lastLayer = networkArchitecture.length - 1;

        for (int layerLevel = lastLayer; layerLevel >= 0; layerLevel--) {
            if (layerLevel == lastLayer) {
                deltas = layers[layerLevel].computeLastLayerDeltas(target);
            } else {
                deltas = layers[layerLevel].computeHiddenLayerDeltas(layers[layerLevel + 1].getPerceptrons(), deltas);
            }
        }
        return createOutputDataForCNN(deltas);
    }

    private double[] createOutputDataForCNN(double[] deltas) {
        Perceptron[] perceptrons = layers[0].getPerceptrons();
        double[] outputData = new double[deltas.length];
        for (int deltaIndex = 0; deltaIndex < deltas.length; deltaIndex++) {
            for (int perceptronIndex = 0; perceptronIndex < deltas.length; perceptronIndex++) {
                outputData[deltaIndex] = perceptrons[perceptronIndex].getWeights()[deltaIndex] * deltas[deltaIndex];
            }
        }
        return outputData;
    }

    private boolean outputIsSameAsTarget(double[] output, double[] target) {
        for (int index = 0; index < target.length; index++) {
            if (abs(output[index] - target[index]) > 0.01) {
//                System.out.println(getClass().getName() + " --> " + "INDEX: " + index + " OUT: " + output[index] + " TARGET: " + target[index] + " ABS: " + abs(output[index] - target[index]));
                return false;
            }
        }
        return true;
    }

    public int[] getNetworkArchitecture() {
        return networkArchitecture;
    }

    public void setNetworkArchitecture(int[] networkArchitecture) {
        this.networkArchitecture = networkArchitecture;
    }

    public Layer[] getLayers() {
        return layers;
    }

    public void setLayers(Layer[] layers) {
        this.layers = layers;
    }


    public class Layer {

        private Perceptron[] perceptrons;

        Layer(int layerSize, int perceptronCapacity) {
            perceptrons = new Perceptron[layerSize];
            for (int index = 0; index < layerSize; index++) {
                perceptrons[index] = new Perceptron(perceptronCapacity);
            }
        }

        public double[] countOutputLayer(double[] inputs) {
            double[] output = new double[perceptrons.length];
            for (int index = 0; index < perceptrons.length; index++) {
                output[index] = perceptrons[index].countOutput(inputs);
            }
            return output;
        }

        public double[] computeHiddenLayerDeltas(Perceptron[] nextLayer, double[] nextLayerDeltas) {
            double[] layerDeltas = new double[perceptrons.length];
            for (int nodeIndex = 0; nodeIndex < perceptrons.length; nodeIndex++) {
                double[] nextLayerWeights = new double[nextLayerDeltas.length];
                for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < nextLayer.length; nextLayerNodeIndex++) {
                    nextLayerWeights[nextLayerNodeIndex] = nextLayer[nextLayerNodeIndex].getWeights()[nodeIndex];
                }
                layerDeltas[nodeIndex] = perceptrons[nodeIndex].countBackPropagationForHiddenLayer(nextLayerDeltas, nextLayerWeights);
            }
            return layerDeltas;
        }

        public double[] computeLastLayerDeltas(double[] target) {
            double[] lastLayerDeltas = new double[perceptrons.length];
            if (perceptrons.length == target.length) {
                for (int index = 0; index < perceptrons.length; index++) {
                    lastLayerDeltas[index] = perceptrons[index].countBackPropagationForOutputLayer(target[index]);
                }
            }
            return lastLayerDeltas;
        }

        public Perceptron[] getPerceptrons() {
            return perceptrons;
        }

    }

}
