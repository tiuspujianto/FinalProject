package processlauncher;


import edu.monash.fit.eduard.grid.Grid;
import edu.monash.fit.eduard.grid.exporter.EsriASCIIGridExporter;
import edu.monash.fit.eduard.grid.importer.EsriASCIIGridImporter;
import edu.monash.fit.eduard.utils.FileUtils;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.exec.CommandLine;
import org.apache.commons.exec.DefaultExecuteResultHandler;
import org.apache.commons.exec.DefaultExecutor;
import org.apache.commons.exec.Executor;
import org.apache.commons.exec.PumpStreamHandler;
import org.apache.commons.exec.ShutdownHookProcessDestroyer;

/**
 * A sample program that loads a grid file, launches a child process, pipes the
 * grid to the child, reads the output grid from the child, and writes the
 * output grid to a file.
 *
 * Uses Apache Commons Exec library for launching the child process. See
 * http://commons.apache.org/proper/commons-exec/
 *
 * @author Bernhard Jenny, Faculty of Information Technology, Monash University,
 * Melbourne, Australia
 */
public class ProcessLauncher {

    //Put on your own directory here pointing towards CUDA exec - IMPORTANT
    private static String DIR_WITH_EXE = "C:\\Users\\aiman\\Desktop\\Monash\\Year 3\\Sem 2\\FIT3162\\Project\\fit3162_gpu_acceleration_cuda_team3\\src\\processlauncher\\CUDA_Executable\\maskFilterAcceleration\\filter_acceleration.cu";

    //Add C CUDA code executable here
    // private static final String EXE_NAME = "processlauncher.ChildProcess";

    // command to launch child process. This command launches a second JVM running a 
    // class file. (The command can instead be path to an exe file). 
    private static String CMD = "nvcc -o compiledcode.out ";

    // path to input grid file. Set to null to show a GUI dialog to select the file
    private static String inputGridFile = null; // "/Users/jennyb/Documents/Java/FIT3161Eduard/data/Gore_Range_Albers_5m.asc";

    // path to output grid file. Set to null to show a GUI dialog to select the file
    private static String outputGridFile = null; // "/Users/jennyb/Documents/Java/FIT3161Eduard/data/out.asc";

    // exit code 0 of child process indicates success
    private static final int SUCCESS_EXIT_CODE = 0;

    public static void main(String[] args) {

        java.awt.EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {
                try {
                    // read input grid file
                    if (inputGridFile == null) {
                        inputGridFile = FileUtils.askFile(null, "Import Grid", true);
                    }
                    Grid inputGrid = EsriASCIIGridImporter.read(inputGridFile);
                    System.out.println("Imported grid from " + inputGridFile);
                    System.out.println(inputGrid.getDescriptionWithStatistics(null));
                    
                    if (DIR_WITH_EXE == null){
                        DIR_WITH_EXE = FileUtils.askFile(null, "Select Your GPU Code", true);
                    }

                    CMD = CMD + DIR_WITH_EXE;

                    // PRINT OUT COMMAND LINE
                    // System.out.println(CMD);

                    CommandLine cmdLine = CommandLine.parse(CMD);
                    Executor executor = new DefaultExecutor();
                    DefaultExecuteResultHandler resultHandler = new DefaultExecuteResultHandler();
                    // end the process when the JVM exits
                    executor.setProcessDestroyer(new ShutdownHookProcessDestroyer());

                    // this exit value is considered a success
                    executor.setExitValue(SUCCESS_EXIT_CODE);

                    // pass size of grid to child process in command line arguments
                    cmdLine.addArgument(Integer.toString(inputGrid.getCols()));
                    cmdLine.addArgument(Integer.toString(inputGrid.getRows()));
                    cmdLine.addArgument(Double.toString(inputGrid.getCellSize()));

                    //Might not be needed
                    cmdLine.addArgument(Double.toString(inputGrid.getNorth()));
                    cmdLine.addArgument(Double.toString(inputGrid.getSouth()));


                    // the child process reads from stdin or cin
                    // GridInputStream streams grid values
                    InputStream childIn = new GridInputStream(inputGrid);
                    
                    // the child process writes the result to stdout or cout
                    ByteArrayOutputStream childOut = new ByteArrayOutputStream();

                    // connect the input and output streams with the child process
                    executor.setStreamHandler(new PumpStreamHandler(childOut, System.err, childIn));

                    // run the child process
                    executor.execute(cmdLine, resultHandler);

                    // block this thread until the child process has finished
                    resultHandler.waitFor();

                    // read result of child process and convert the result to a new grid            
                    if (executor.isFailure(SUCCESS_EXIT_CODE) == false) {

                        // allocate output grid
                        // Copies the size of the input grid
                        Grid outputGrid = Grid.shallowCopy(inputGrid);

                        // read result from childOut
                        byte[] b = childOut.toByteArray();

                        for (int row = 0, nRows = outputGrid.getRows(); row < nRows; row++) {
                            for (int col = 0, nCols = outputGrid.getCols(); col < nCols; col++) {
                                int i = (col + row * nCols) * 4;
                                // convert four bytes to one float
                                // this assumes big-endian order, shuffle bytes if in little-endian order
                                int b1 = b[i] & 0xFF;
                                int b2 = b[i + 1] & 0xFF;
                                int b3 = b[i + 2] & 0xFF;
                                int b4 = b[i + 3] & 0xFF;                                
                                int intBits = b4 << 24 | b3 << 16 | b2 << 8 | b1;
                                float f = Float.intBitsToFloat(intBits);
                                outputGrid.setValue(f, col, row);
                            }
                        }

                        // write output grid
                        if (outputGridFile == null) {
                            outputGridFile = FileUtils.askFile(null, "Export Grid", false);
                        }
                        if (outputGridFile != null) {
                            EsriASCIIGridExporter.export(outputGrid, outputGridFile, null);
                            System.out.println("Exported grid to " + outputGridFile);
                        }
                    } else {
                        System.out.println("Fail");
                    }

                } catch (Throwable ex) {
                    Logger.getLogger(ProcessLauncher.class.getName()).log(Level.SEVERE, null, ex);
                } finally {
                    System.exit(0);
                }
            }
        });
    }
}
