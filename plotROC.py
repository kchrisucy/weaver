# Example: python plotROC.py output/mlp_predict_test.root,output/mlp_predict_ep3.root --outputDir plots --logY

import uproot3 as uproot
#import uproot
import ROOT
import numpy
import array
import os
import ctypes
import sys
from optparse import OptionParser

def GetGraph(x, y, xerrl, xerrh, yerrl, yerrh, name):
    graph = ROOT.TGraphAsymmErrors(len(x), array.array("d",x), array.array("d",y),
                                   array.array("d",xerrl), array.array("d",xerrh),
                                   array.array("d",yerrl), array.array("d",yerrh))
    graph.SetName(name)
    return graph

def CreateLegend(xmin=0.55, ymin=0.75, xmax=0.85, ymax=0.85):
    leg = ROOT.TLegend(xmin, ymin, xmax, ymax)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.040)
    leg.SetTextFont(42)
    leg.SetLineColor(1)
    leg.SetLineStyle(1)
    leg.SetLineWidth(1)
    leg.SetFillColor(0)
    return leg

def ApplyStyle(h, color, line=ROOT.kSolid):
    h.SetLineColor(color)
    h.SetMarkerColor(color)
    h.SetMarkerStyle(8)
    h.SetMarkerSize(0.5)
    h.SetLineStyle(line)
    h.SetLineWidth(3)
    h.SetTitle("")
    return

def SavePlot(canvas, saveDir, saveName, saveFormats=["pdf"], verbose=False):
    
    savePath = "%s/%s" % (saveDir, saveName)    
    for ext in saveFormats:
        fileName = "%s.%s" % (savePath, ext)
        canvas.SaveAs( fileName )
    return


def CalcEfficiency(histo_s, histo_b):
    '''
    Calculate the signal and background efficiency vs DNN score
    '''
    # Initialize sigma variables
    nbins    = histo_s.GetNbinsX()
    sigmaAll = ctypes.c_double(0.0)
    sigmaSel = ctypes.c_double(0.0)
    All_s    = histo_s.IntegralAndError(0, nbins+1, sigmaAll, "")
    All_b    = histo_b.IntegralAndError(0, nbins+1, sigmaAll, "")
    eff_s    = []
    eff_b    = [] 
    xvalue   = []
    error    = []
    
    # For-loop: All bins
    for i in range(0, nbins+1):
        Sel_s = histo_s.IntegralAndError(i, nbins+1, sigmaSel, "")
        Sel_b = histo_b.IntegralAndError(i, nbins+1, sigmaSel, "")

        if (All_s <= 0):
            All_s = 1
            Sel_s = 0
        if (All_b <= 0):
            All_b = 1
            Sel_b = 0

        eff_s.append(Sel_s/All_s)
        eff_b.append(Sel_b/All_b)
        error.append(0)
        xvalue.append(histo_s.GetBinCenter(i))
    
    #print "%d: %s" % (len(xvalue), xvalue)
    #print "%d: %s" % (len(eff_s), eff_s)
    return xvalue, eff_s, eff_b, error

def CreateScoreHistogram(name, xMin, xMax, nBins):
    histo = ROOT.TH1F(name, name, nBins, xMin, xMax)
    histo.GetXaxis().SetTitle("score")
    histo.GetYaxis().SetTitle("a.u")
    return histo

def PlotOutput(histo_s, histo_b, saveDir, saveFormats):
    
    ROOT.gStyle.SetOptStat(0)

    # Create canvas
    canvas = ROOT.TCanvas()
    canvas.cd()
    canvas.SetLogy()

    if 1:
        histo_s.Scale(1./histo_s.Integral())
        histo_b.Scale(1./histo_b.Integral())

    ymax = max(histo_s.GetMaximum(), histo_b.GetMaximum())

    ApplyStyle(histo_s, ROOT.kBlue)
    ApplyStyle(histo_b, ROOT.kRed)

    for h in [histo_s,histo_b]:
        h.SetMinimum(100)
        h.SetMaximum(ymax*1.1)
        h.GetXaxis().SetTitle("Output")
        h.GetYaxis().SetTitle("Entries")
        h.Draw("HIST SAME")
    
    # Create legend
    leg=CreateLegend(0.6, 0.75, 0.9, 0.85)
    leg.AddEntry(histo_s, "signal","l")
    leg.AddEntry(histo_b, "background","l")
    leg.Draw()
    
    saveName = histo_s.GetName().split("/")[-1].replace("_sig","").replace(".root","")
    
    SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return

def PlotROC(graphList, logY, saveDir, saveFormats):
    '''
    Plot the ROC curves
    '''
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas()
    canvas.cd()
    if logY:
        canvas.SetLogy()
    leg = CreateLegend(0.15, 0.75, 0.45, 0.9)
    lineStyle = ROOT.kSolid
    saveName = "ROC"

    # For-loop: All graphs
    for i, gr in enumerate(graphList, 0):
        gr_name = gr.GetName()        
        color = i+2 # Skip white

        ApplyStyle(gr, color, lineStyle)
        gr.SetMarkerSize(0)
        gr.GetXaxis().SetTitle("Signal Efficiency")
        gr.GetYaxis().SetTitle("Background Efficiency")
        gr.GetXaxis().SetRangeUser(0.0, 1.0)
        gr.GetYaxis().SetRangeUser(0.0, 1.0)
        if i == 0:
            gr.Draw("apl")
        else:
            gr.Draw("pl same")

        leg.AddEntry(gr, gr_name, "l")

        if (i > 1):
            leg.Draw("same")
        else:
            leg.Draw()

    SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return

def getROC(fileName, outputDir):
    #fileName = "output/mlp_predict_test.root"
    print("Opening root file: %s" % fileName)
    #f         = ROOT.TFile.Open(fileName)
    #tree      = f.Get("Events")
    #variables = tree.GetListOfBranches()
    
    # Definitions
    brNames = ["is_signal_new","score_is_signal_new", "is_bkg", "score_is_bkg"]
    xMin  = 0.0
    xMax  = 1.0
    nBins = 100

    myBins  = numpy.linspace(xMin, xMax, nBins)
    #uRootFile = uproot.open(fileName)["Events"]
    uRootFile = uproot.open(fileName)["tree"]

    # Get dataframes with selected branch names
    df        = uRootFile.pandas.df(brNames)
    df_sig    = df[df['is_signal_new'] == 1]
    df_bkg    = df[df['is_signal_new'] == 0]

    # Digitize returns the bin number that each entry belongs to
    digi_sig = numpy.digitize( numpy.clip(df_sig["score_is_signal_new"].values, xMin, xMax), bins=myBins, right=False )
    digi_bkg = numpy.digitize( numpy.clip(df_bkg["score_is_signal_new"].values, xMin, xMax), bins=myBins, right=False )

    #print(df_bkg["is_signal_new"].values)

    # Create signal and background histograms
    histo_sig = CreateScoreHistogram(fileName+"_sig", xMin, xMax, nBins)
    histo_bkg = CreateScoreHistogram(fileName+"_bkg", xMin, xMax, nBins)

    for i in range(nBins):
        value = histo_sig.GetBinCenter(i) # value: the value of bin i
        sig_i = numpy.sum(digi_sig == i)  # sig_i: the number of signal entries in bin i
        bkg_i = numpy.sum(digi_bkg == i)  # bkg_i: the number of background entries in bin i 
        histo_sig.Fill(value,sig_i)
        histo_bkg.Fill(value,bkg_i)

    PlotOutput(histo_sig, histo_bkg, outputDir, ["pdf"])

    # Calculate signal and background efficiency vs output
    xvalue, eff_s, eff_b, error = CalcEfficiency(histo_sig, histo_bkg)
    #Get ROC curve (signal efficiency vs bkg efficiency)
    graph_roc = GetGraph(eff_s, eff_b, error, error, error, error, histo_sig.GetName().replace("_sig",""))
        
    return graph_roc
    
#==============================================================================================================================
if __name__ == "__main__":

# Setup the parser options object
    parser = OptionParser(usage="Usage: %prog [options]" , add_help_option=False,conflict_handler="resolve")

    parser.add_option("--outputDir", dest="outputDir", type="string", default="plots", 
                      help="Name of the output directory")       

    parser.add_option("--logY", dest="logY", action="store_true", default=False, 
                      help="Plot y-axis (exlusion limit) as logarithmic")

    (opts, parseArgs) = parser.parse_args()

if len(sys.argv) < 2:
    print ("Please provide a valid input root file")
    sys.exit(0)

fileNames = sys.argv[1]
if "," in sys.argv[1]:
    fileNames = fileNames.split(",")
else:
    fileNames = [fileNames]

rocCurves = []
for f in fileNames:
    g = getROC(f, opts.outputDir)
    rocCurves.append(g)
    
# Create output directory
if not os.path.exists(opts.outputDir):
    os.mkdir(opts.outputDir)
    
PlotROC(rocCurves, opts.logY, opts.outputDir, ["pdf"]) 
