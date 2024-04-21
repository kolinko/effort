/*
 
 Main.
 
 Overall code structure of other files:

 - runNetwork - main inference loop

 - model - computations like rmsnorm, also definitions of Vector/Matrix class

 - loader - holds layer information and loads it up
 
 - bucketMul - main multiplication algorithm
 
 - convert - converts from .safetensors into the bucketized version.
             you'll probably better off just getting the bucketed weights from HF
 
 */


import os
import Foundation
import Metal
import simd

let args = CommandLine.arguments
let runMode = args.count > 1 ? args[1] : ""

var serverReady = false
let gpu = Gpu()
print("\nEffort Engine v.0.0.2 BETA")

//runConvert([.mistral, .fp16])
// ^ uncomment to run conversion for other models.
//   be sure to check the convert script comments before


/*
 
 below should be refactored into a Conf class.
 Need to do it smartly though, because during testing of larger models you want to easily
 be able to load fewer layers / fewer experts to pass by tests
 
 */

let stateDim = 4096
let hiddenDim = 14336
let goQ8 = false
assert(!goQ8, "Q8 not implemented fully yet!")
var percentLoad = autoAdjustPercent(max: goQ8 ? 0x8 : 0x10)
                  // a number from 0x00 to 0x10 for FP16, or 0x0 to 0x8.
                  // not really percent, need to make design decision here - either switch to real percent
                  // or stick with the current convention but make it clear why.



var numLayers = 32
var numExperts = 1
var numTokens = 30

let goMistral = numExperts == 1

testSetup("models", "mistral", "buckets-FP16.safetensors.index.json")
let modelData = Model(numLayers: numLayers, numExperts: numExperts, percentLoad: percentLoad)
let t = Tokeniser(modelData)

if runMode != "quickstart" && runMode != "--no-benchmark" {
    goQuickBucketPerformance()
}

let headDim = 128  // Example head dimension
let numHeadsKV = 8
let numHeads = 32
let kvRepeats : Int = numHeads/numHeadsKV
let maxSeqLen = 2048
let maxTokens = maxSeqLen
let freqsCis = createFreqsCis2(headDim: headDim, maxSeqLen: maxSeqLen)

print()
if runMode != "quickstart" {
    print("»»» How are ", terminator: "")
    _ = runNetwork(tokens: t.embed([1, 1602, 460]), effort:1.0)
}
// ^ quick test to see if the model works and how well. super useful during development.


numTokens = 150

var effort: Double = 1.0

switch runMode {
    
    case "playground":
        goPlayground()
    case "quiz":
        goQuiz()
    case "benchmark":
        goBenchmarkSimilarity()
    case "bucket":
        goBucketPerformance()
    default:
        break
    
}

var prevQuery : String? = nil

while true {
    print("This is a test environment. Doesn't hold context!")
    print("Enter 0-100 to change Effort, or type in query to see the output.")
    while true {
        print("> ", terminator: "")
        if let input = readLine() {
            if let number = Int(input), (0...100).contains(number) {
                effort = Double(number)/100.0
                if prevQuery != nil {
                    let tokens = t.embed("<s>[INST]\(prevQuery!)[/INST]")
                    _ = runNetwork(tokens: tokens, effort:effort)
                }
            } else if input == "r" {
                // a nice simple test case
                let tq = "What's larger - Radom, Poland, or Sydney, Australia?"
                print("? \(tq)")
                let tokens = t.embed("<s>[INST]\(tq)[/INST]")
                _ = runNetwork(tokens: tokens, effort:effort, srcTokenIds: encode(prompt:"<s>[INST]\(tq)[/INST]"))
            } else {
                prevQuery = input
                let tokens = t.embed("<s>[INST]"+input+"[/INST]")
                _ = runNetwork(tokens: tokens, effort:effort)
            }
        }
    }
}
