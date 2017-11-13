package SW504;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Program {
    //Height Width and channels of pictures
    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 3;
    //Number of possible labels = number of outputs
    protected static int numLabels = 5;
    //Batchsize is the number of pictures loaded in memory at runtime
    protected static int batchSize = 32;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int iterations = 1;
    protected static int epochs = 15;
    //Paths for test and training sets
    protected static File trainPath  =  new File("C:/Users/palmi/Desktop/NeuralTesting/trainSet");
    protected static File testPath = new File("C:/Users/palmi/Desktop/NeuralTesting/testSet");
    private static FileSplit trainData = new FileSplit(trainPath, NativeImageLoader.ALLOWED_FORMATS,rng);
    private static FileSplit testData = new FileSplit(testPath,NativeImageLoader.ALLOWED_FORMATS,rng);
    protected static boolean trainWithTransform = false;
    protected static MultiLayerNetwork trainedNetwork = lenetModel();
    private static Logger log = LoggerFactory.getLogger(Program.class);
    private static ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();

    public static void main(String[] args) throws IOException {

        earlyStopTraining();



        /*trainedNetwork = trainNetwork_withoutTransformation(recordReader,trainData);

        if(trainWithTransform){
            ImageTransform flipTransform1 = new FlipImageTransform(rng);
            ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
            ImageTransform warpTransform = new WarpImageTransform(rng, 42);

            List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});

            trainedNetwork = trainNetwork_withTransformation(recordReader,transforms,trainData,trainedNetwork);
        }

        evaluateNetwork(recordReader,new ImagePreProcessingScaler(0,1),testData,trainedNetwork);

        saveNetwork("trained_model.zip",trainedNetwork);
        */
    }

    private static void UiServerSetup(){
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        trainedNetwork.setListeners(new StatsListener(statsStorage));
    }

    private static MultiLayerNetwork  trainNetwork_withTransformation(ImageRecordReader recordReader, List<ImageTransform> transforms, InputSplit trainSet, MultiLayerNetwork neuralNetwork) throws IOException {
        DataSetIterator dataIterator;
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        MultipleEpochsIterator trainIter;


        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainSet, transform);
            dataIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIterator);
            dataIterator.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, dataIterator);
            neuralNetwork.fit(trainIter);
        }

        return neuralNetwork;
    }

    private static MultiLayerNetwork trainNetwork_withoutTransformation(ImageRecordReader recordReader, InputSplit trainSet) throws IOException {
        recordReader.initialize(trainSet);
        recordReader.setListeners(new LogRecordListener());

        DataSetIterator dataIterator = new RecordReaderDataSetIterator(recordReader,batchSize,1,numLabels);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIterator);
        dataIterator.setPreProcessor(scaler);

        trainedNetwork.init();

        for(int i = 0; i< epochs; i++){
            System.out.println("Running " + i);
            trainedNetwork.fit(dataIterator);
        }

        return trainedNetwork;
    }

    private static void saveNetwork(String fileName, MultiLayerNetwork neuralNetwork) throws IOException {
        File saveTo = new File(fileName);
        ModelSerializer.writeModel(neuralNetwork,saveTo,false);
    }

    private static void evaluateNetwork(ImageRecordReader recordReader, DataNormalization scaler, InputSplit fileSet, MultiLayerNetwork neuralNetwork) throws IOException {
        recordReader.reset();
        recordReader.initialize(fileSet);
        DataSetIterator testIte = new RecordReaderDataSetIterator(recordReader,batchSize,1,numLabels);
        scaler.fit(testIte);
        testIte.setPreProcessor(scaler);

        Evaluation eval = new Evaluation(numLabels);

        while(testIte.hasNext()){
            DataSet next = testIte.next();
            INDArray output = neuralNetwork.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(),output);
        }
        log.info(eval.stats());
    }

    private static SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private static void earlyStopTraining() throws IOException {

        ImageRecordReader validationReader = new ImageRecordReader(height,width,channels,labelGenerator);
        ImageRecordReader trainingReader = new ImageRecordReader(height,width,channels,labelGenerator);

        validationReader.initialize(testData);
        trainingReader.initialize(trainData);

        DataSetIterator validationSet = new RecordReaderDataSetIterator(validationReader,batchSize,1, numLabels);
        DataSetIterator trainingSet = new RecordReaderDataSetIterator(trainingReader,batchSize,1,numLabels);

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(epochs))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(30, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(validationSet,true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver("trained_model.zip"))
                .build();

        EarlyStoppingTrainer esTrainer = new EarlyStoppingTrainer(esConf,lenetConf(),trainingSet);

        EarlyStoppingResult result = esTrainer.fit();

        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        //Print score vs. epoch
        Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
        List<Integer> list = new ArrayList<Integer>(scoreVsEpoch.keySet());
        Collections.sort(list);
        System.out.println("Epoch vs. Score:");
        for( Integer i : list){
            System.out.println(i + "\t" + scoreVsEpoch.get(i));
        }
    }

   /* public static MultiLayerNetwork inceptionModel(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.9))
                .activation(Activation.RELU)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)


    }
*/
    public static MultiLayerConfiguration lenetConf(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .learningRate(0.0045) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2,2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        return conf;
    }

    public static MultiLayerConfiguration alexnetConf(){
        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(0.9))
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2)
                .biasLearningRate(1e-2*2)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .list()
                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[]{3,3}))
                .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[]{3,3}))
                .layer(6,conv3x3("cnn3", 384, 0))
                .layer(7,conv3x3("cnn4", 384, nonZeroBias))
                .layer(8,conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[]{3,3}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        return conf;
    }

    public static MultiLayerNetwork lenetModel() {
        /**
         * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2,2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public static MultiLayerNetwork alexnetModel() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(0.9))
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2)
                .biasLearningRate(1e-2*2)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .list()
                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(2, maxPool("maxpool1", new int[]{3,3}))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
                .layer(5, maxPool("maxpool2", new int[]{3,3}))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(6,conv3x3("cnn3", 384, 0))
                .layer(7,conv3x3("cnn4", 384, nonZeroBias))
                .layer(8,conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[]{3,3}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    private static DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

}
