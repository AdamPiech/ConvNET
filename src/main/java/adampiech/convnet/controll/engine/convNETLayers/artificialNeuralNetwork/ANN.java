package adampiech.convnet.controll.engine.convNETLayers.artificialNeuralNetwork;

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

    public double[] process(double[] inputs) {
        for (int layerLevel = 0; layerLevel < networkArchitecture.length; layerLevel++) {
            if (layers[layerLevel] == null) {
                generateNetworkLayers(inputs, layerLevel);
            } else {
                System.out.println("NOT NULL " + layerLevel);
            }
        }
        return layers[networkArchitecture.length - 1].createOutput(); //temporary
    }

    private void generateNetworkLayers(double[] inputs, int layerLevel) {
        if (layerLevel == 0) {
            layers[layerLevel] = new Layer(inputs, networkArchitecture[layerLevel]);
        } else {
            layers[layerLevel] = new Layer(layers[layerLevel - 1].createOutput(), networkArchitecture[layerLevel]);
        }
    }

    private class Layer {

        public Perceptron[] perceptrons;

        Layer(double[] inputs, int layerSize) {
            generateLayer(inputs, layerSize);
        }

        private void generateLayer(double[] inputs, int layerSize) {
            perceptrons = new Perceptron[layerSize];
            for (int index = 0; index < layerSize; index++) {
                perceptrons[index] = new Perceptron(inputs);
            }
        }

        public double[] createOutput() {
            double[] output = new double[perceptrons.length];
            for (int index = 0; index < perceptrons.length; index++) {
                output[index] = perceptrons[index].generateOutput();
            }
            return output;
        }
    }

}
