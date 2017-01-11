package adampiech.convnet.controll.engine.convNETLayers.artificialNeuralNetwork;

import java.util.Random;

import static java.lang.Math.exp;

/**
 * Created by Adam Piech on 2016-11-07.
 */
public class Perceptron {

    private final static double ETA = -1.0;

    private Random rand = new Random();

    private int perceptronCapacity;
    private double[] inputs;
    private double output;
    private double[] weights;
    private double bias;

    public Perceptron(int perceptronCapacity) {
        this.perceptronCapacity = perceptronCapacity;
        generateWeights(perceptronCapacity);
        generateBias();
    }

    public double countOutput(double[] inputs) {
        this.inputs = inputs;
        double output = 0.0;
        for(int index = 0; index < inputs.length; index++) {
            output += inputs[index] * weights[index];
        }
        return this.output = activationFunction(output + bias);
    }

    private double activationFunction(double value) {
        return 1 / (1 + exp(-value));
    }

    public double countBackPropagationForOutputLayer(double target) {
        double delta = output * (1 - output) * (output - target);
        countNewWeights(delta);
        countNewBias(delta);
        return delta;
    }

    public double countBackPropagationForHiddenLayer(double[] nextLayerDelta, double[] nextLayerWeights) {
        double sumNextLayerParams = 0.0;
        for(int index = 0; index < nextLayerDelta.length; index++) {
           sumNextLayerParams += nextLayerDelta[index] * nextLayerWeights[index];
        }
        double delta = output * (1 - output) * sumNextLayerParams;
        countNewWeights(delta);
        countNewBias(delta);
        return delta;
    }

    private void countNewWeights(double delta) {
        for(int index = 0; index < perceptronCapacity; index++) {
            weights[index] = weights[index] + ETA * delta * inputs[index];
        }
    }

    private double countNewBias(double delta) {
        return bias + ETA * delta;
    }

    private void generateWeights(int quantity) {
        weights = new double[quantity];
        for(int index = 0; index < weights.length; index++) {
//            weights[index] = rand.nextDouble() * 2.0 - 1.0;
            weights[index] = rand.nextGaussian();
        }
    }

    private void generateBias() {
//        bias = rand.nextDouble();
        bias = 1.0;
    }

    public int getPerceptronCapacity() {
        return perceptronCapacity;
    }

    public void setPerceptronCapacity(int perceptronCapacity) {
        this.perceptronCapacity = perceptronCapacity;
    }

    public double[] getInputs() {
        return inputs;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

}
