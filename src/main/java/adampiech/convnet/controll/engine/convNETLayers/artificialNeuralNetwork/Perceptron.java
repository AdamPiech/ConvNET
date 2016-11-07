package adampiech.convnet.controll.engine.convNETLayers.artificialNeuralNetwork;

import java.util.Random;

import static java.lang.Math.exp;

/**
 * Created by Adam Piech on 2016-11-07.
 */
public class Perceptron {

    private Random rand = new Random();

    private double[] inputs;
    private double[] weights;
    private double bias;

    public Perceptron(double[] inputs) {
        this.inputs = inputs;
        generateWeights(inputs.length);
        generateBias();
    }

    public double generateOutput() {
        double output = 0.0;
        for(int index = 0; index < inputs.length; index++) {
            output += inputs[index] * weights[index];
        }
        return activationFunction(output + bias);
    }

    private double activationFunction(double value) {
        return 1 / (1 + exp(-value));
    }

    private void generateWeights(int size) {
        weights = new double[size];
        for (double weight : weights) {
            weight = rand.nextDouble();
        }
    }

    private void generateBias() {
        bias = rand.nextDouble();
    }

    public double[] getInputs() {
        return inputs;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
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
