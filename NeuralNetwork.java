/*
* Author: Linda Zeng
* Date Modified from A-B-C-D with Back Propagation to N-Layer with Back Propagation: November 13, 2025
* 
* This module implements a trainable, runnable N-layer neural network optimized with
* backpropagation for steepest descent.
* The program:
*   1) Reads user configuration from a control file, echoes it, and initializes network arrays,
*   2) (Optionally) trains using steepest descent and reports training results,
*   3) Runs the network on test cases and reports results.
* Training and running modes are independent.
* This program supports loading the weights from and saving the weights to binary files
* specified by the user.
*
* Parsing from files is implemented using classes from the java.util package, including
* Java’s Properties class along with the Boolean, Integer, and Double classes.
*
* Table of Contents:
*
* public boolean setConfigFromFile()                                      
* public double randomize(double lower, double upper)
* public double sigmoidFunc(double x)
* public double sigmoidFuncDeriv(double x)
* public double hyperbolicTanFunc(double x)
* public double hyperbolicTanFuncDeriv(double x)
* public double activationFunc(double x)
* public double activationFuncDeriv(double x)
* public void setCalculated()
* public void echoConfig()
* public void allocateArrays()
* public void populateWeightArraysRand()
* public boolean populateInputArrayFromFile() 
* public void populateInputArrayManually() 
* public boolean populateTruthArrayFromFile()
* public void populateTruthArrayManually() 
* public boolean populateArrays()
* public void setUpOneTrainingCase(int t)   
* public double runOneCaseForTraining()
* public void trainOneCase()
* public void trainNetwork() 
* public void reportTrainingResults()
* public void setUpOneTestCase(int t)
* public void runOneTestCase()
* public void saveOneTestCase(int t)   
* public void runNetwork()
* public void reportResults()
* public void saveWeightsToFile(String filename) 
* public boolean loadWeightsFromFile(String filename)
* public static void main(String[] args)
*/
import java.util.Date;
import java.io.*;
import java.util.Properties;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;


public class NeuralNetwork           
{
/*
* The following constants define the indices of the input layer and
* first hidden layer for an N-layer neural network implemented with zero-indexed arrays,
* as well as the name of the default configuration file.
*/
   public final int INPUT_LAYER = 0;                     
   public final int FIRST_HIDDEN_LAYER = 1;   
   public final String DEFAULT_CONFIG_FILE = "configuration.txt";

/*
* The following variables indicate the indices of the last hidden layer
* and output layer for an N-layer neural network implemented with zero-indexed arrays.
*/
   public int lastHiddenLayer;   
   public int outputLayer;

/*
* The following variables are associated with the number of layers in the network.
*/
   public int nLayers;
   public int nActivationLayers;
   public int nHiddenLayers;
   
/*
* The following variables are associated with the number of elements in each activation layer.
*/
   public int[] nActivations;

/*
* The following variables store the values in the network's elements and the network weights.
*/
   public double[][] a;
   public double[][][] w;

/* 
* The following variables store values used for steepest descent in training.
*/
   public double[][] theta;
   public double[][] psi;                           

/* 
* The following variables are associated with the input table.
*/
   public int numCases;
   public double[][] inputTable;
   public boolean printInputTable; 
   public String typeInputInit;             // Takes on "loaded" or "manual" which determines if inputs are read from a file or not

/* 
* The following variables are associated with the truth table.
*/
   public boolean haveTruthTable;           // Indicates if a truth table exists
   public boolean printTruthTable;          // Determines if the truth table is printed after running
   public String typeTruthInit;             // Takes on "loaded" or "manual" which determines if truth table is read from a file
   public String truthTableName;            // Name of truth table, used for echo statements and manually setting the truth table 
   public double[][] truthTable;            // Variable storing truth table values
   public double[] T;                       // The truth values for the current training case

/* 
* The following variable is associated with the model outputs.
*/
   public double[][] outputs;               // Stores the model's outputs for each test case for echoing purposes

/* 
* The following variables specify user preferences about setup, training, and saving weights to a file.
*/
   public String typeWeightInit;            // Takes on "random" or "loaded from file" as its value
   public static boolean trainMode;
   public static boolean weightSaving;

/*
* The following variables specify the names of text files containing configuration details, input, and truth table, 
* and binary files containing the weights.
*/
   public static String configFileName;    // Name of the file that contains the configuration details
   public String inputFileName;            // Name of the file that contains the input table, used if typeInputInit is "loaded" 
   public String truthFileName;            // Name of the file that contains the truth table, used if typeTruthInit is "loaded" 
   public static String loadFileName;      // Name of the file that contains weights to load
   public static String saveFileName;      // Name of the file to save weights to

/* 
* The following variables refer to runtime training parameters specified by the user.
*/
   public double minRand;
   public double maxRand;
   public double learningFactor;
   public int maxIterations;
   public double errorThreshold;
   public int keepAlive;         

/*
* The following variables store values which dynamically update during the training iterations.
*/
   public int trainingIterations;    // current number of training iterations during training
   public double iterationError;     // current iteration error during training

/*
* The following variables are associated with recording the time elapsed of training and running.
*/
   public static long timeElapsedTrain;
   public static long timeElapsedRun;

/*
* Initializes the neural network with the default configuration file name.
*/
   public NeuralNetwork()
   {
      configFileName = DEFAULT_CONFIG_FILE;    
   }

/*
* Sets user configuration variables by reading from a control file.
* The control file includes information about the network
* configuration, the user's preferences about 
* setup, training, and echoing, the user's specified
* runtime training parameters, and configuration of input and truth tables. 
* Returns true if configuration variables are successfully loaded; false otherwise.
*/
   public boolean setConfigFromFile()
   {
      boolean successful = true;
      Properties props = new Properties();   

      try (FileInputStream fis = new FileInputStream(configFileName))
      {
         props.load(fis);                 

/* 
* Theis variable reflects the network's connectivity layers.
*/
         nLayers = Integer.parseInt(props.getProperty("nLayers"));  
            
/* 
* This block reads in the user-specified number of activations in each layer and checks that
* the correct quantity of activation layer sizes were given. The number of activation layers should be
* one more than the number of connectivity layers the user passed in.
*/
         String[] nActivationsFile = props.getProperty("nActivations").split(",");
         nActivationLayers = nLayers + 1;                      // Defined in order to use for reading in the number of activations      

         if (nActivationsFile.length != nActivationLayers)     // Checks size of nActivationsFile is consistent 
         {                                      
            throw new Exception("The size of 'nActivations' is mismatched with the number of layers passed in.");
         }

         nActivations = new int[nActivationLayers];            // initializes nActivations                                          

         for (int n = 0; n < nActivationLayers; n++)           // populates nActivations                                      
         {
            nActivations[n] = Integer.parseInt(nActivationsFile[n].trim());      
         }           

/* 
* These variables reflect configuration details of the input table.
*/
         numCases = Integer.parseInt(props.getProperty("numCases"));
         printInputTable = Boolean.parseBoolean(props.getProperty("printInputTable"));
         typeInputInit = props.getProperty("typeInputInit");
         inputFileName = props.getProperty("inputFileName");  

/* 
* These variables reflect configuration details of the truth table.
*/
         haveTruthTable = Boolean.parseBoolean(props.getProperty("haveTruthTable"));
         printTruthTable = Boolean.parseBoolean(props.getProperty("printTruthTable"));
         truthTableName = props.getProperty("truthTableName"); 
         typeTruthInit = props.getProperty("typeTruthInit");
         truthFileName = props.getProperty("truthFileName");                        

/* 
* These variables reflect user preferences about setup, training, and saving weights.
*/
         trainMode = Boolean.parseBoolean(props.getProperty("trainMode"));
         typeWeightInit = props.getProperty("typeWeightInit");          
         weightSaving = Boolean.parseBoolean(props.getProperty("weightSaving"));
         loadFileName = props.getProperty("loadFileName");        
         saveFileName = props.getProperty("saveFileName");

/* 
* These variables reflect runtime training information.
*/
         minRand = Double.parseDouble(props.getProperty("minRand"));    
         maxRand = Double.parseDouble(props.getProperty("maxRand"));    
         learningFactor = Double.parseDouble(props.getProperty("learningFactor"));
         maxIterations = Integer.parseInt(props.getProperty("maxIterations")); 
         errorThreshold = Double.parseDouble(props.getProperty("errorThreshold"));   
         keepAlive = Integer.parseInt(props.getProperty("keepAlive"));   

         if (keepAlive < 1)  // Ensures the user set keepAlive to a valid value
         {
            keepAlive = 0;
         }
      } // try (FileInputStream fis = new FileInputStream(configFileName))
      catch (Exception e)
      {
         System.out.println("Error when reading configuration file: " + e);
         successful = false;
      }

      return successful;
   } // public void setConfigFromFile()

/*
* Takes in two doubles representing the lower and upper bound,
* and generates a random double
* between the specified range,
* inclusive of the lower bound 
* and exclusive of the upper bound.
*/
   public double randomize(double lower, double upper)
   {
      return Math.random() * (upper - lower) + lower; 
   } 

/*
* Defines the sigmoid activation function.
* Takes in an input double x, applies the
* sigmoid function to the input and returns
* the output as a double.
*/
   public double sigmoidFunc(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

/*
* Defines the derivative of the sigmoid activation
* function. Takes in an input double x, 
* applies the derivative
* of the sigmoid function to x and 
* returns the output as a double.
*/
   public double sigmoidFuncDeriv(double x)
   {
      double sigmoidFuncValue = sigmoidFunc(x);
      return sigmoidFuncValue * (1.0 - sigmoidFuncValue);
   } // public double sigmoidFuncDeriv(double x)

/*
* Defines the hyperbolic tangent activation function.
* Takes in an input double x, applies the
* hyperbolic tangent function to the input and returns
* the output as a double.
*/
   public double hyperbolicTanFunc(double x)
   {
      double epsilon = (x < 0.0) ? 1.0 : -1.0;
      double eEpsilon2x = Math.exp(epsilon * 2.0 * x);  
      return epsilon * (eEpsilon2x - 1.0) / (eEpsilon2x + 1.0);
   } // public double hyperbolicTanFunc(double x)

/*
* Defines the derivative of the hyperbolic tangent
* activation function.
* Takes in an input double x, applies the derivative of the
* hyperbolic tangent function to the input and returns
* the output as a double.
*/
   public double hyperbolicTanFuncDeriv(double x)
   {
      double hyperbolicTanValue = hyperbolicTanFunc(x);
      return 1.0 - hyperbolicTanValue * hyperbolicTanValue;
   } // public double hyperbolicTanFuncDeriv(double x)

/*
* Defines the activation function
* used for each element in the neural
* network. Takes in x, the input to the function,
*  and returns the output
* of the function as a double
*/
   public double activationFunc(double x)
   {
      return sigmoidFunc(x);
   }

/*
* Defines the derivative of activation function
* used for each element in the neural
* network. Takes in x, the input to the function,
*  and returns the output
* of the function as a double
*/
   public double activationFuncDeriv(double x)
   {
      return sigmoidFuncDeriv(x);
   }

/*
* Calculates and sets helpful variables 
* using the user's defined configuration
* variables.
*/
   public void setCalculated()
   {
      nActivationLayers = nLayers + 1;
      nHiddenLayers = nLayers - 1;

      outputLayer = nActivationLayers - 1;    // The output layer index is set assuming implementation using zero-indexed arrays
      lastHiddenLayer = outputLayer - 1;                    
   } // public void setCalculated()

/*
* Prints the user configuration variables
* including the network configuration,
* the user's preferences regarding setup,
* training, and echoing, the runtime training 
* parameters, and information about the
* data table and truth table used.
*/
   public void echoConfig()
   {
/*
* Prints variables related to network configuration
*/
      System.out.println("Number of Layers: " + nLayers);  
      System.out.print("Network Configuration: ");

      for (int n = 0; n < nActivationLayers - 1; n++)  // Print number of activations in every layer except the last layer                  
      {
         System.out.print(nActivations[n] + "-");                                                          
      }

      int n = nActivationLayers - 1; 
      System.out.println(nActivations[n] + "\n");      // Print number of activations in the last layer            

/*
* Prints variables related to configuration details and echoing preferences for input table
*/
      System.out.println("Number of Input Cases: " + numCases);
      System.out.println("Printing the Input Table: " + printInputTable);   
      System.out.println("Type of Input Table Population: " + typeInputInit);

      if (typeInputInit.equals("loaded"))
      {
         System.out.println("Loading the inputs from text file (name: " + inputFileName + ")");      
      }

      if (typeInputInit.equals("binary"))
      {
         System.out.println("Loading the inputs from directory (name: " + inputFileName + ")");      
      }
      System.out.println();    

/*
* Prints variables related to configuration details and echoing preferences for truth table
*/
      System.out.println("Truth table is included: " + haveTruthTable);

      if (haveTruthTable)
      {
         System.out.println("Name of Truth Table: " + truthTableName);
         System.out.println("Printing the Truth Table: " + printTruthTable);
         System.out.println("Type of Truth Table Population: " + typeInputInit);

         if (typeTruthInit.equals("loaded"))
         {
            System.out.println("Loading the truth values from text file (name: " + truthFileName + ")");      
         }
      } // if (haveTruthTable)
      System.out.println();

/*
* Prints variables related to user preferences relating to setup and training
*/
      System.out.println("Training the network: " + trainMode);
      System.out.println("Type of Weight Initialization: " + typeWeightInit);

      if (typeWeightInit.equals("loaded from file"))
      {
         System.out.println("Loading the weights from binary file (name: " + loadFileName + ")");      
      }

/*
* Prints variables related to user preferences relating to saving weights.
*/
      if (weightSaving)
      {
         System.out.println("Saving the weights to binary file (name: " + saveFileName + ")");
      }
      else
      {
         System.out.println("Not saving the weights to binary file");
      }
      System.out.println();

/*
* Prints runtime training requirements only if training is set to true
*/
      if (trainMode)         
      {
         System.out.println("Runtime Training Requirements:");

/* 
* Prints upper and lower bound of random numbers if random weight initialization is true
*/
         if (typeWeightInit.equals("random"))     
         { 
            System.out.println("Randomizing Weights in the Range [" + minRand + ", " + maxRand + ")");
         }

         System.out.println("Learning Factor: " + learningFactor);
         System.out.println("Maximum number of allowed training iterations: " + maxIterations);
         System.out.println("Error Threshold: " + errorThreshold);
         System.out.println("Keep Alive: " + keepAlive);
         System.out.println(); 
      } // if (trainMode) 
   } // public void echoConfig()

/*
* Allocates space for all arrays
* for running and, if training is enabled,
* for all arrays for training.
*/
   public void allocateArrays()
   {
/* 
* Allocates space for the network's values and weights
*/
      a = new double[nActivationLayers][];  
      w = new double[nLayers][][];                    
      
      for (int n = INPUT_LAYER; n < outputLayer; n++)               // Iterate over all activation layers except the output activation layer              
      {
         a[n] = new double[nActivations[n]];                          
         w[n] = new double[nActivations[n]][nActivations[n + 1]];  // Weights are associated to the left activation layer they connect to                                           
      }

      int n = outputLayer;                                                 
      a[n] = new double[nActivations[n]];                          // Final output activation layer

/*
* If training, allocates space for arrays needed for training
*/
      if (trainMode)
      {
         theta = new double[nActivationLayers - 1][];       
         psi = new double[nActivationLayers][];

         for (n = FIRST_HIDDEN_LAYER; n < outputLayer; n++)     // Iterate over all hidden activation layers                   
         {
            theta[n] = new double[nActivations[n]];             // Just the hidden layers for theta
            psi[n + 1] = new double[nActivations[n + 1]];       // From the second hidden layer to the output layer (inclusive) for psi                                            
         }

         haveTruthTable = true;                                 // Flag needed to allocate the truth table for training
         T = new double[nActivations[n]];                       
      } // if (trainMode)

/* 
* Allocates space for input tables, truth tables, and model outputs
*/
      n = INPUT_LAYER;
      inputTable = new double[numCases][nActivations[n]];

      n = outputLayer;
      if (haveTruthTable)
      {
         truthTable = new double[numCases][nActivations[n]];                 
      } // if (haveTruthTable)

      outputs = new double[numCases][nActivations[n]];                                               
   } // public void allocateArrays()

/*
* Initializes weights
* with random values between
* the user specified range.
*/
   public void populateWeightArraysRand()
   {
      for (int n = 0; n < nLayers; n++)
      {
         for (int k = 0; k < nActivations[n]; k++)                               
         {
            for (int j = 0; j < nActivations[n + 1]; j++)
            {
               w[n][k][j] = randomize(minRand, maxRand);
            }
         } // for (int k = 0; k < nActivations[n]; k++)    
      } // for (int n = 0; n < nLayers; n++)
   } // public void populateWeightArraysRand()

/*
* Populates input data table by reading
* in from a file and parsing the file's values. Works
* for any number of cases and inputs as long
* as the number of cases and inputs in the file
* matches the configuration. If they do not match,
* the method returns false, indicating that
* loading inputs was unsuccessful. 
*/
   public boolean populateInputArrayFromFile() 
   {
      boolean successful = true;
      Properties props = new Properties();   
      
      try (FileInputStream fis = new FileInputStream(inputFileName))
      {
         props.load(fis); 

         int n = INPUT_LAYER;       

/*
* Checks to see if the number of test cases and number of inputs in the file matches the current configuration
*/
         int numCasesFile = Integer.parseInt(props.getProperty("numCases"));
         int nInputsFile = Integer.parseInt(props.getProperty("nInputs"));

         if ((numCasesFile != numCases) || (nInputsFile != nActivations[n]))
         {
            throw new Exception("The size of the input table in the file does not match current configuration.");
         }

/*
* Parses the input table from the file and populates the inputTable array with the corresponding values     
*/
         for (int t = 0; t < numCases; t++)
         {
            for (int m = 0; m < nActivations[n]; m++)
            {
               inputTable[t][m] = Double.parseDouble(props.getProperty("inputTable." + t).split(",")[m]);                
            }
         } // for (int t = 0; t < numCases; t++)
      } // try (FileInputStream fis = new FileInputStream(inputFileName))
      catch (Exception e)
      {
         System.out.println("Error when reading input table: " + e);
         successful = false;
      }

      return successful;
   } // public boolean populateInputArrayFromFile() 

/*
* Populates input data table by reading
* in from a BINARY file and parsing the file's values. Works
* for any number of cases and inputs as long
* as the number of cases and inputs in the file
* matches the configuration. If they do not match,
* the method returns false, indicating that
* loading inputs was unsuccessful. 
*/
   public boolean populateInputArrayFromBinary(String dirPath) 
   {
    boolean successful = true;

    String extension = ".act";
    final int n = INPUT_LAYER;
    final int expectedInputs = nActivations[n];
    final long expectedBytes = (long) expectedInputs * 8L; // doubles = 8 bytes

    try {
        // 1) Collect and sort files
        List<Path> files = Files.list(Paths.get(dirPath))
                .filter(p -> p.getFileName().toString().endsWith(extension))
                .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                .collect(Collectors.toList());

        if (files.isEmpty()) {
            throw new FileNotFoundException("No " + extension + " files found in: " + dirPath);
        }

        // 2) Make numCases match file count (each file = one case)
        numCases = files.size();

        // 3) Allocate input table if needed / not sized correctly
        if (inputTable == null || inputTable.length != numCases || inputTable[0].length != expectedInputs) {
            inputTable = new double[numCases][expectedInputs];
        }

        // 4) Read each file into one row of inputTable
        for (int t = 0; t < numCases; t++) {
            Path p = files.get(t);

            long size = Files.size(p);
            if (size != expectedBytes) {
                throw new IOException("File size mismatch for " + p.getFileName()
                        + ": got " + size + " bytes, expected " + expectedBytes
                        + " (expected " + expectedInputs + " doubles).");
            }

            try (DataInputStream dis = new DataInputStream(
                    new BufferedInputStream(Files.newInputStream(p)))) {

                for (int m = 0; m < expectedInputs; m++) {
                    inputTable[t][m] = dis.readDouble(); // assumes writeDouble() format
                }
            }
        }

    } catch (Exception e) {
        System.out.println("Error when reading binary input cases: " + e);
        successful = false;
    }

    return successful;
   } // public boolean populateInputArrayFromBinary() 

/*
* Manually populates input
* data table with two 
* inputs for each case.
* Works only for four case.
*/
   public void populateInputArrayManually() 
   {
      inputTable[0][0] = 0.0;   
      inputTable[0][1] = 0.0;

      inputTable[1][0] = 0.0;
      inputTable[1][1] = 1.0;

      inputTable[2][0] = 1.0;
      inputTable[2][1] = 0.0;

      inputTable[3][0] = 1.0;
      inputTable[3][1] = 1.0;
   } // public void populateInputArrayManually()

/*
* Populates truth table by reading
* in from a file and parsing the file's values. Works
* for any number of cases and inputs as long
* as the number of cases and inputs in the file
* matches the configuration. If they do not match,
* the method returns false, indicating that
* loading the truth table was unsuccessful. 
*/
   public boolean populateTruthArrayFromFile()
   {
      boolean successful = true;
      Properties props = new Properties();   
      
      try (FileInputStream fis = new FileInputStream(truthFileName))
      {
         props.load(fis);     

         int n = outputLayer;

/*
* Checks to see if the number of test cases and number of outputs in the file matches the current configuration
*/
         int numCasesFile = Integer.parseInt(props.getProperty("numCases"));
         int nOutputsFile = Integer.parseInt(props.getProperty("nOutputs"));
         String truthTableNameFile = props.getProperty("truthTableName");

         if ((numCasesFile != numCases) || (nOutputsFile != nActivations[n]) || !(truthTableNameFile.equals(truthTableName))) 
         {
            throw new Exception("The truth table in the file does not match current configuration.");
         }

/*
* Parses the truth table from the file and populates the truthTable array with the corresponding values     
*/
         for (int t = 0; t < numCases; t++)
         {
            for (int i = 0; i < nActivations[n]; i++)
            {
               truthTable[t][i] = Double.parseDouble(props.getProperty("truthTable." + t).split(",")[i]);                    
            }
         } // for (int t = 0; t < numCases; t++)
      } // try (FileInputStream fis = new FileInputStream(inputFileName))
      catch (Exception e)
      {
         System.out.println("Error when reading truth table: " + e);
         successful = false;
      }

      return successful;
   } // public boolean populateTruthArrayFromFile()

/*
* Manually populates the truth table with output values
* corresponding to either three logical operations
* (AND, OR, and XOR, in that order) or a single AND operation.
* The number of outputs depends on the value of truthTableName.
* This function assumes four input cases.
*/
   public void populateTruthArrayManually()                
   {
      if (truthTableName.equals("AND OR XOR"))
      {
         truthTable[0][0] = 0.0;
         truthTable[0][1] = 0.0;
         truthTable[0][2] = 0.0;

         truthTable[1][0] = 0.0;
         truthTable[1][1] = 1.0;
         truthTable[1][2] = 1.0;

         truthTable[2][0] = 0.0;
         truthTable[2][1] = 1.0;
         truthTable[2][2] = 1.0;

         truthTable[3][0] = 1.0;
         truthTable[3][1] = 1.0;
         truthTable[3][2] = 0.0;
      } // if (truthTableName.equals("AND OR XOR"))
      else
      {
         truthTable[0][0] = 0.0;
         truthTable[1][0] = 0.0;
         truthTable[2][0] = 0.0;
         truthTable[3][0] = 1.0;
      }
   } // public void populateTruthArrayManually()

/*
* Populates data, truth, and weight arrays,
* initializing the weight arrays based
* on the user's preference, which is random, fixed,
* or loaded in from a file. Returns true if the
* arrays were populated successfully,
* which occurs if no errors were run into
* while loading weights, inputs, and truth table.
*/
   public boolean populateArrays()
   {
      boolean populatedSuccessfully = true;

/*
* Populates weight arrays depending on the user's preference
*/
      if (typeWeightInit.equals("random"))
      {
         populateWeightArraysRand();
      }
      else
      {
         populatedSuccessfully = loadWeightsFromFile(loadFileName);
      }

/*
* Populates input arrays depending on the user's preference
*/
      if (typeInputInit.equals("manual"))
      {
         populateInputArrayManually();
      }
      else if (typeInputInit.equals("binary"))
      {
         populateInputArrayFromBinary(inputFileName);
      }
      else
      {
         populatedSuccessfully &= populateInputArrayFromFile();
      }
      
/*
* If the truth table exists, populates truth table based on the user's preference
*/
      if (haveTruthTable)
      {
         if (typeInputInit.equals("manual"))
         {
            populateTruthArrayManually();
         }
         else
         {
            populatedSuccessfully &= populateTruthArrayFromFile();
         }
      } // if (haveTruthTable)

      return populatedSuccessfully;
   } // public void populateArrays()

/*
* Takes in the index of the current training case
* and inserts the input and truth values corresponding
* with the current training case into the activations 
* layer and truth values array. Requires that a truth table exists.
*/
   public void setUpOneTrainingCase(int t)          
   {
      int n = INPUT_LAYER;                   
      for (int m = 0; m < nActivations[n]; m++)
      {
         a[n][m] = inputTable[t][m];
      }

      n = outputLayer;
      for (int i = 0; i < nActivations[n]; i++)
      {
         T[i] = truthTable[t][i];
      }
   } // public void setUpOneTrainingCase(int t)

/*
* Executes one forward pass of the network
* for a single training case. Calculates the
* activations of the hidden and output layers
* based on the current weights and input values,
* and collects information needed for training.
* Returns the error for one train case.
*/
   public double runOneCaseForTraining()
   {
      for (int n = FIRST_HIDDEN_LAYER; n < outputLayer; n++)         // Iterates over all hidden activation layers
      {
         for (int j = 0; j < nActivations[n]; j++)
         {
            theta[n][j] = 0.0;                            
            for (int k = 0; k < nActivations[n - 1]; k++)
            {
               theta[n][j] += a[n - 1][k] * w[n - 1][k][j];
            }

            a[n][j] = activationFunc(theta[n][j]);     
         } // for (int j = 0; j < nActivations[n]; j++)
      } // for (int n = FIRST_HIDDEN_LAYER; n < outputLayer; n++)

      int n = outputLayer;                                          // Performs calculations at the output activation layer
      double errorOneCase = 0.0;                                    // Zero the error accumulator
      for (int i = 0; i < nActivations[n]; i++)
      {
         double thetai = 0.0;                         
         for (int j = 0; j < nActivations[n - 1]; j++)
         {
            thetai += a[n - 1][j] * w[n - 1][j][i];
         }

         a[n][i] = activationFunc(thetai);

         double omegai = T[i] - a[n][i];                                
         psi[n][i] = omegai * activationFuncDeriv(thetai);
         errorOneCase += omegai * omegai;
      } // for (int i = 0; i < nActivations[n]; i++)

      return errorOneCase / 2.0;
   } // public void runOneCaseForTraining()

/*
* Adjusts the weights for the connectivity layers using
* information collected during the forward evaluation 
* for training, as part of back propagation and
* the steepest descent learning rule.
*/
   public void trainOneCase()
   {
      for (int n = lastHiddenLayer; n > FIRST_HIDDEN_LAYER; n--)        // Iterates from last hidden layer to second hidden layer  
      {
         for (int k = 0; k < nActivations[n]; k++)  
         {
            double omega = 0.0;
            for (int j = 0; j < nActivations[n + 1]; j++)    
            {
               omega += psi[n + 1][j] * w[n][k][j];                               
               w[n][k][j] += learningFactor * a[n][k] * psi[n + 1][j];       
            } // for (int j = 0; j < nActivations[n + 1]; j++)   

            psi[n][k] = omega * activationFuncDeriv(theta[n][k]);     
         } // for (int k = 0; k < nActivations[n]; k++)  
      } // for (int n = lastHiddenLayer; n > FIRST_HIDDEN_LAYER; n--)

      int n = FIRST_HIDDEN_LAYER;                             
      for (int m = 0; m < nActivations[n]; m++)                        // Performs necessary computations for the first hidden layer
      {        
         double omegam = 0.0;
         for (int k = 0; k < nActivations[n + 1]; k++)  
         {
            omegam += psi[n + 1][k] * w[n][m][k];
            w[n][m][k] += learningFactor * a[n][m] * psi[n + 1][k];     
         } // for (int j = 0; j < nActivations[n + 1]; j++)    
         
         double psim = omegam * activationFuncDeriv(theta[n][m]);         
                                                      
         for (int x = 0; x < nActivations[n - 1]; x++)                  // Update weights associated with the input layer
         {
            w[n - 1][x][m] += learningFactor * a[n - 1][x] * psim;    
         } // for (int m = 0; m < nActivations[n - 1]; m++)    
      } // for (int k = 0; k < nActivations[n]; k++) 
   } // public void trainOneCase()

/*
* Trains the network through backpropagation
* by running the network through each test case,
* updating the weights through steepest descent,
* and repeating this process iteratively until reaching
* the maximum iterations or error threshold
* specified by the user. If keep-alive messages are turned
* on, the method echoes keep-alive messages to the user.
*/
   public void trainNetwork() 
   {
      trainingIterations = 0;
      do // while (trainingIterations < maxIterations || iterationError < errorThreshold)
      {
         double totalError = 0.0;                
         for (int t = 0; t < numCases; t++)                
         {
            setUpOneTrainingCase(t);                         
            totalError += runOneCaseForTraining(); 
            trainOneCase(); 
         } // for (int t = 0; t < numCases; t++)  

         iterationError = totalError / (double)numCases;    // Calculates and updates the average error for the current iteration
         trainingIterations++; 
         
         if ((keepAlive != 0) && ((trainingIterations % keepAlive) == 0))    
         {
            System.out.printf("Iteration %d, Error = %f\n", trainingIterations, iterationError);
         }
      } while (trainingIterations < maxIterations && iterationError > errorThreshold);
   } // public void trainNetwork()

/*
* Prints the training results, including
* the reason for exiting training, the
* training iterations reached, 
* the average error reached, and time elapsed.
*/
   public void reportTrainingResults()
   {
      System.out.println("Training Exit Information:");

/* 
* Prints the stopping condition reached
*/
      if (trainingIterations == maxIterations)
      {
         if (iterationError <= errorThreshold)     // Checks if both stopping conditions are reached
         {
            System.out.println("Training iterations reached max iterations and average error was less than threshold error");
         }
         else                                   
         {
            System.out.println("Training iterations reached maximum iterations");
         }
      } // if (trainingIterations == maxIterations)
      else
      {
         System.out.println("Average error was less than the threshold error");
      }

/* 
* Prints the final training values reached from training
*/
      System.out.println("Iterations reached: " + trainingIterations);
      System.out.println("Average error reached: " + iterationError);

/* 
* Prints time elapsed for training the network
*/
      System.out.println("Time Elapsed for Training: " + timeElapsedTrain +  " milliseconds");
      System.out.println();
   } // public void reportTrainingResults()

/*
* Takes in the index of the test case
* and places the input data corresponding with
* the test case into the input activation layer.
*/
   public void setUpOneTestCase(int t)
   {
      int n = INPUT_LAYER;
      for (int m = 0; m < nActivations[n]; m++)
      {
         a[n][m] = inputTable[t][m];
      }
   } // public void setUpOneTestCase(int t)

/*
* Runs the network with its current weights on one test case
* Calculates the network, assuming the inputs are in
* the input activation layer.
*/
   public void runOneTestCase()
   {
      for (int n = FIRST_HIDDEN_LAYER; n <= outputLayer; n++)     // All hidden activation layers and the output layer
      {
         for (int j = 0; j < nActivations[n]; j++)
         {
            double thetaj = 0.0;                             
            for (int k = 0; k < nActivations[n - 1]; k++)
            {
               thetaj += a[n - 1][k] * w[n - 1][k][j];
            }

            a[n][j] = activationFunc(thetaj);     
         } // for (int j = 0; j < nActivations[n]; j++)
      } // for (int n = FIRST_HIDDEN_LAYER; n <= outputLayer; n++)
   } // public void runOneTestCase()

/*
* Takes in the index of one test case,
* and saves the outputs of the corresponding test case,
* for the purposes of later printing.
*/
   public void saveOneTestCase(int t)                   
   {
      int n = outputLayer;
      for (int i = 0; i < nActivations[n]; i++)   
      {
         outputs[t][i] = a[n][i];                                                                                                       
      }   
   } // public void saveOneTestCase(int t)
 
/*
* Runs the network through all of the test cases
* and saves outputs into an array for echoing purposes.
*/
   public void runNetwork()
   { 
      for (int t = 0; t < numCases; t++)   
      {
         setUpOneTestCase(t);
         runOneTestCase();                                                                                         
         saveOneTestCase(t);              
      } // for (int t = 0; t < numCases; t++)   
   } // public void runNetwork()

/*
* Prints the results of running the network,
* including the input and truth table if the user
* specified for them to be printed, the network
* outputs for each test case, and the time elapsed.
*/
   public void reportResults()
   {
/* 
* Prints entries in input table (optional)
*/
      if (printInputTable)
      {
         System.out.println("Input Table:"); 
          
         int n = INPUT_LAYER;
         for (int t = 0; t < numCases; t++)   
         {                    
            for (int m = 0; m < nActivations[n]; m++)
            {
               System.out.print(inputTable[t][m] + " "); // Prints all inputs for one test case on one row
            } 
            System.out.println();                        // Prints new line for each test case            
         } // for (int t = 0; t < numCases; t++)   
      } // if (printInputTable)

/*
* Prints entries in truth table (optional)
*/
      if (haveTruthTable && printTruthTable)     
      {
         System.out.println("Truth Table (" + truthTableName + "):");  

         int n = outputLayer;
         for (int t = 0; t < numCases; t++)   
         {  
            for (int i = 0; i < nActivations[n]; i++)
            {
               System.out.print(truthTable[t][i] + " "); // Prints all truth values for one test case one on one row
            } 
            System.out.println();                        // Prints new line for each test case                     
         } // for (int t = 0; t < numCases; t++)   
      } // if (haveTruthTable && printTruthTable)     
      
/* 
* Prints outputs for each case (always) 
*/
      System.out.println("Outputs:"); 

      int n = outputLayer;
      for (int t = 0; t < numCases; t++)   
      {
         for (int i = 0; i < nActivations[n]; i++)
         {
            System.out.printf("%-25.4f", outputs[t][i]);  // Prints outputs to 4 decimal places in 25-character column
         }   
         System.out.println();                                    // Prints new line for each test case                    
      } // for (int t = 0; t < numCases; t++)  

/* 
* Prints time elapsed for running the network
*/
      System.out.println("Time Elapsed for Running: " + timeElapsedRun + " milliseconds");
      System.out.println();
   } // public void reportResults()

/*
* Saves the network’s weights to a binary file.
* Writes the number of layers, the number of 
* activations in each activation layer, and
* the weights of each connectivity layer.
* The name of the binary file is provided as input.  
*/
   public void saveWeightsToFile(String filename)                                    
   {
      try (DataOutputStream dataOutput = new DataOutputStream(new FileOutputStream(filename)))   
      { 
         dataOutput.writeInt(nLayers); 

         for (int n = 0; n < nActivationLayers; n++)  // Iterate over activation layers 
         {
            dataOutput.writeInt(nActivations[n]);             
         }     

         for (int n = 0; n < nLayers; n++)           // Iterate over connectivity layers 
         {
            for (int k = 0; k < nActivations[n]; k++)                               
            {
               for (int j = 0; j < nActivations[n + 1]; j++)
               {
                  dataOutput.writeDouble(w[n][k][j]);  
               }
            } // for (int k = 0; k < nActivations[n]; k++)    
         } // for (int n = 0; n < nLayers; n++)
      } // try (DataOutputStream dataOutput = new DataOutputStream(new FileOutputStream(filename)))     
      catch (Exception e)
      {
         System.out.println("Error when saving weights: " + e);    
         System.out.println("Continuing program without saving weights...\n");  
      }
   } // public void saveWeightsToFile(String filename)

/*
* Loads network configuration and weights from a binary file. Verifies that the file’s
* metadata (nLayers, nInputs, nHidden1, nHidden2, nOutputs) matches the current network; if so,
* populates the weight arrays and returns true. Returns false on any read
* error or configuration mismatch.
* The name of the binary file must be provided as input.
*/
   public boolean loadWeightsFromFile(String filename)
   {
      boolean loadedSuccessful = true;      

      try (DataInputStream dataInput = new DataInputStream(new FileInputStream(filename)))       
      { 
/*
* Checks the number of layers in the network configuration in the weights file, and throws an error if
* this configuration is mismatched with the current network configuration.
*/
         int nLayersFile = dataInput.readInt(); 

         if (nLayersFile != nLayers)
         {
            throw new Exception("Current network configuration is mismatched with configuration in the weights file.");    
         }

/*
* Checks the number of activations in each activation layer in the network configuration in the weights file, and throws an error if
* this configuration is mismatched with the current network configuration.
*/   
         for (int n = 0; n < nActivationLayers; n++)
         {
            if (nActivations[n] != dataInput.readInt())          
            {
               throw new Exception("Current network configuration is mismatched with configuration in the weights file.");
            }
         } // for (int n = 0; n < nActivationLayers; n++) 

/*
* Iterates over connectivity layers and populates weights from the input file.
*/   
         for (int n = 0; n < nLayers; n++)              
         {
            for (int k = 0; k < nActivations[n]; k++)                               
            {
               for (int j = 0; j < nActivations[n + 1]; j++)
               {
                  w[n][k][j] = dataInput.readDouble();
               }
            } // for (int k = 0; k < nActivations[n]; k++)    
         } // for (int n = 0; n < nLayers; n++)
      } // try (DataInputStream dataInput = new DataInputStream(new FileInputStream(filename)))      
      catch (Exception e)
      {
         System.out.println("Error when loading weights: " + e);  
         loadedSuccessful = false;       
      }

      return loadedSuccessful;
   } // public void loadWeightsFromFile(String filename)

/* 
* This is the main entry point for the neural network program.
*
* It includes setting up user configuration parameters and echoing them to confirm the setting,
* allocating and initializing arrays for the network, optionally training the neural network
* and reporting training results, optionally saving the weights, and running the network
* on test cases and reporting test results.
* It also records and displays the start and end time of the run.
*/
   public static void main(String[] args)
   {
      Date startDate = new Date();
      System.out.println("Starting: " + startDate);
      System.out.println("==========================================");

      NeuralNetwork nn = new NeuralNetwork();               // Initializes the neural network with default configuration file     

      if (args.length > 0)                                  // Checks if there are any command line arguments passed in
      {
         configFileName = args[0];                          // Overrides default configuration filename with filename passed in   
      } 

      System.out.println("Reading configuration details from: " + configFileName);              

      boolean successfulConfig = nn.setConfigFromFile();                                 

      if (successfulConfig)                                 // Checks if configuration was read without errors to continue the program
      {
         nn.setCalculated(); 
         nn.echoConfig();    
         nn.allocateArrays(); 
         boolean successfulPopulation = nn.populateArrays();    

         if (successfulPopulation)                         // Checks if arrays were populated without errors to continue the program
         {
            if (trainMode)
            {
               System.out.println("Began training");
               long startTimeTrainMillis = System.currentTimeMillis();         
               nn.trainNetwork();   
               timeElapsedTrain = System.currentTimeMillis() - startTimeTrainMillis;  

               nn.reportTrainingResults();   
            } // if (trainMode)

            if (weightSaving)
            {
               nn.saveWeightsToFile(saveFileName);
            }

            long startTimeRunMillis = System.currentTimeMillis();
            nn.runNetwork();  
            timeElapsedRun = System.currentTimeMillis() - startTimeRunMillis;  

            nn.reportResults();
         } // if (successfulPopulation)
         else
         {
            System.out.println("The arrays were not populated successfully, so the program has been exited.");
         }
      } // if (successfulConfig)
      else
      {
         System.out.println("The configuration was not loaded successfully, so the program has been exited.");
      }

      Date endDate = new Date();
      System.out.println("Finished: " + endDate);
   } // public static void main(String[] args)
} // public class NeuralNetwork