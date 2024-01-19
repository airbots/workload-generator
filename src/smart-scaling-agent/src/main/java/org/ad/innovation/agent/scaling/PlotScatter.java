//
//import java.awt.Color;
//import java.io.File;
//import java.io.IOException;
//
//import javax.swing.JPanel;
//
//import org.jfree.chart.ChartFactory;
//import org.jfree.chart.ChartPanel;
//import org.jfree.chart.ChartRenderingInfo;
//import org.jfree.chart.ChartUtilities;
//import org.jfree.chart.JFreeChart;
//import org.jfree.chart.axis.NumberAxis;
//import org.jfree.chart.plot.PlotOrientation;
//import org.jfree.chart.plot.XYPlot;
//import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
//import org.jfree.chart.servlet.ServletUtilities;
//import org.jfree.data.xy.XYDataset;
//import org.jfree.data.xy.XYSeries;
//import org.jfree.data.xy.XYSeriesCollection;
//import org.jfree.ui.ApplicationFrame;
//
///**
// * A demo scatter plot.
// */
//public class PlotScatter extends ApplicationFrame {
//	static String xCoordinate="X";
//	static String yCoordinate="Y";
//    /**
//     * A demonstration application showing a scatter plot.
//     *
//     * @param title  the frame title.
//     * @return
//     */
//
//	public static void saveChartAsPNG(JFreeChart chart, int width, int height
//			  ) throws IOException {
//			  if (chart == null) {
//			  throw new IllegalArgumentException("Null 'chart' argument.");
//			  }
//			  String prefix = ServletUtilities.getTempFilePrefix();
//			  File tempFile=new File("file.jpg");
//
//			  ChartUtilities.saveChartAsPNG(tempFile, chart, width, height);
//			}
//
//
//    public PlotScatter(String title, XYSeries xys,String x, String y) throws IOException {
//        super(title);
//        xCoordinate=x;
//        yCoordinate=y;
//        JPanel chartPanel = createDemoPanel(xys);
//        chartPanel.setPreferredSize(new java.awt.Dimension(400, 300));
//        setContentPane(chartPanel);
//
//    }
//
//    private static JFreeChart createChart(XYDataset dataset) throws IOException {
//        JFreeChart chart = ChartFactory.createScatterPlot("Distribution Verification",
//                xCoordinate, yCoordinate, dataset, PlotOrientation.VERTICAL, true, false, false);
//
//        XYPlot plot = (XYPlot) chart.getPlot();
//        plot.setNoDataMessage("NO DATA");
//        plot.setDomainZeroBaselineVisible(true);
//        plot.setRangeZeroBaselineVisible(true);
//
//
//        XYLineAndShapeRenderer renderer
//                = (XYLineAndShapeRenderer) plot.getRenderer();
//        renderer.setSeriesOutlinePaint(0, Color.black);
//        renderer.setUseOutlinePaint(true);
//        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
//        domainAxis.setAutoRangeIncludesZero(false);
//        domainAxis.setTickMarkInsideLength(2.0f);
//        domainAxis.setTickMarkOutsideLength(0.0f);
//
//        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
//        rangeAxis.setTickMarkInsideLength(2.0f);
//        rangeAxis.setTickMarkOutsideLength(0.0f);
//
//
//        //save the chart
//        //saveChartAsPNG(chart,800,600);
//
//        return chart;
//    }
//
//    /**
//     * Creates a panel for the demo (used by SuperDemo.java).
//     *
//     * @return A panel.
//     * @throws IOException
//     */
//    public static JPanel createDemoPanel(XYSeries xys) throws IOException {
//        XYDataset data = new XYSeriesCollection(xys);
//        JFreeChart chart = createChart(data);
//        ChartPanel chartPanel = new ChartPanel(chart);
//        //chartPanel.setVerticalAxisTrace(true);
//        //chartPanel.setHorizontalAxisTrace(true);
//        // popup menu conflicts with axis trace
//        chartPanel.setPopupMenu(null);
//
//        chartPanel.setDomainZoomable(true);
//        chartPanel.setRangeZoomable(true);
//        return chartPanel;
//    }
//
//    /**
//     * Starting point for the demonstration application.
//     *
//     * @param args  ignored.
//     */
//
//}