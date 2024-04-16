//
//  benchmark.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 13/04/2024.
//

import Foundation

func testABCD(_ _query: String) {
    let query = "<s>[INST]\(_query)[/INST] The answer is number:"
    let logits = [28740, 28750, 28770, 28781]
    let embeded = t.embed(query)
    for effort in [1, 0.5, 0.4, 0.3, 0.25, 0.22, 0.20, 0.15, 0.10, 0.08] {
        let reply = runNetwork(tokens: embeded, effort:effort, limitLogits: logits).reply
        print(reply)
        print("\(Int(effort*100))% -> \(reply)")
    }
}

func verifyABCD(_ _query: String, answer: Int, scale: [Double]) -> [Bool] {
    let query = "<s>[INST]\(_query)[/INST] The answer is number: "
    let logits = [28740, 28750, 28770, 28781]
    let embeded = t.embed(query)
    var outputs = [Bool]()
    for effort in scale {
        let reply = runNetwork(tokens: embeded, effort:effort, limitLogits: logits).reply
        let score = Int(reply) == answer
        outputs.append(score)
    }
    return outputs
}

func makeScale() -> [Double] {
    var scale = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35]
    for i in 0...15 {
        scale.append(0.3-Double(i)*0.02)
    }
    var out = "Scale: ["
    for s in scale {
        out += "\(s*100, precision: 0), "
    }
    print(out+"]")
    return scale

}

func goQuiz() {
    print("Testing BoolQ")
    let scale = makeScale()
    print("Scale: ", scale)
    let qa = loadQuiz()

    numTokens = 800
    
    var results = [[Bool]]()
    var count = 0
    for _test in qa {
        var test = _test
        count += 1
        print("Testing QA, \(count) of \(qa.count)")
        var prompt = "Answer this question: \"\(test.prompt)\". Choices:"
        test.shuffle()
        for i in 0..<test.choices.count {
            prompt += "\(i+1). \(test.choices[i])"
        }
        print(test.prompt)
        let out = verifyABCD(prompt, answer: test.answer+1 , scale: scale)
        print(out)
        results.append(out)
        
        
        var result = Array(repeating: 0, count: scale.count)
        for r in results {
            for s in 0..<scale.count {
                result[s] += r[s] ? 1 : 0
            }
        }
        
        for s in 0..<scale.count {
            print("\(Int(scale[s]*100))% -> \(result[s]); ", terminator: "")
        }
        print()
    }
    
}


struct ItemQuiz: Decodable {
    let prompt: String
    var choices: [String]
    var answer: Int
    
    mutating func shuffle() {
            var newChoices = choices
            var lastIndex = choices.count - 1
            while lastIndex > 0 {
                let randomIndex = Int.random(in: 0...lastIndex)
                newChoices.swapAt(lastIndex, randomIndex)
                if answer == lastIndex {
                    answer = randomIndex
                } else if answer == randomIndex {
                    answer = lastIndex
                }
                lastIndex -= 1
            }
            choices = newChoices
        }
}


func loadQuiz() -> [ItemQuiz] {
    let fileUrl = URL(fileURLWithPath: "./benchmarks/data/quiz.json")
    let jsonData = try! Data(contentsOf: fileUrl)
    let items = try! JSONDecoder().decode([ItemQuiz].self, from: jsonData)
    return items
}



func goBenchmarkSimilarity() {
    let scale = makeScale()

    numTokens = 500
//    let query = "[INST]Write a condensed perl implementations of the following: Dijkstra's algorithm, text search, quicksort, llama llm inference, brainfuck interpreter. No comments, just write the code. Include data loaders. Make it perl-golf style.[/INST]"
  let query = "[INST]Write an extremely long and convoluted story about potatoes[/INST]"

    let tokIds = encode(prompt: query)
    let baselineText = runNetwork(tokens: t.embed(tokIds), effort: 1).reply

//    let tokenIds2 = baseline.hitMiss
    let embeded = t.embed(query+baselineText)
    let control = runNetwork(tokens: embeded, effort: 1, returnPredictions: true).hitMiss
    
    print(control.count, "tokens prompt.")
    
    for effort in scale {
        let test = runNetwork(tokens: embeded, effort: effort, returnPredictions: true).hitMiss
        var num = 0
        for i in 0..<control.count {
            num += control[i] == test[i] ? 1 : 0
        }
        let res = Double(100*Float(num)/Float(control.count))
        print("\(Int(effort*100))% -> " + "\(res, precision: 2)%")
        
    }
  
    exit(0)
}


func goBucketPerformance() {
    let scale = makeScale()
    let modelData = Model(numLayers: 32, numExperts: 1, percentLoad: 0x10)
    let t = Tokeniser(modelData)
    
    let v = t.embed([1])[0]// random token embeded to get a state vector
    
    let ew = modelData.layers[10]!.wq
    
    let control = VectorFloat(shape:[ew.outSize])
    
    basicMul(v: v, by: ew.core!, out: control)
    expertMul(v: v, by: ew, out: control, effort: 1.0)
    
    for s in scale {
        let test = VectorFloat(shape:[ew.outSize])
        expertMul(v: v, by: ew, out: test, effort: s)
        print("\(s, perc: ()) -> \(test.cosineSimilarityTo(control), precision: 5)")
    }
    
    print("speed comparison")

    
    /*
     
         1. Warmups are within timeIt.
         2. Comparing 3xWK (3x4096x4096) vs 1xW1 (4x14336) here.
            W2 and WQ/K/V are not optimised in this version, bc they need to have their parameters fixed -
            see bucketMulFast deployment comment in bucketMul.swift.
     
     */

    // warmups are within timeIt
    timeIt(repeats: 10000) { i in
        basicMul(v: v, by: modelData.layers[i % 32]!.wq.core!, out: control)
        basicMul(v: v, by: modelData.layers[i % 32]!.wq.core!, out: control)
        basicMul(v: v, by: modelData.layers[i % 32]!.wq.core!, out: control)
    }
    print("^ MPS")

    
    for s in scale {
        let test = VectorFloat(shape:[ew.outSize])
        timeIt(repeats: 10000) { i in
            expertMul(v: v, by: modelData.layers[i % 32]!.w1, out: test, effort: s)
        }
        expertMul(v: v, by: ew, out: test, effort: s)
        print("^ BucketMul (\(s, perc: ())")
    }

    
    exit(0)
    
   // bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: control, effort: 1.0)

    
    
    //    let control = basicMul(v: v, by: ew.core!) //
    let test = VectorFloat(shape:[ew.outSize])
    
    /*
    print(v.str)
    print(ew.buckets.str)
    print()
    print(control.str)
    //gpu.startCapture()
    for q in [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0] {
        expertMul(v: v, by: ew, out: test, effort: q)
        assert(!test.hasNan)
        //gpu.stopCapture()
        //        print()
        let score = test.cosineSimilarityTo(control)
        print("\(Int(q*100))%: \(Double(score), precision:5)", score>0.99 ? "✓" : "✗")
        //        print()
    }*/

    let q = 1.0
    bucketMulFaster(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: q)
    let score = test.cosineSimilarityTo(control)
    print("\(Int(q*100))%: \(Double(score), precision:5)", score>0.99 ? "✓" : "✗")

    exit(0)
    
    
    /*expertMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: control)
//    gpu.startCapture()
    bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: 1)
    gpu.stopCapture()

    timeIt(repeats:1000) { i in
        let ew = modelData.layers[i % 3]!.w1
        expertMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: 1)
    }
    print()
    timeIt(repeats:1000) { i in
        let ew = modelData.layers[i % 3]!.w1
        bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: 1)
    }

    //    gpu.startCapture()

    bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: 1)
    gpu.stopCapture()

    print()
    let score = test.cosineSimilarityTo(control)
    print("\(Double(score), precision:5)", score>0.99 ? "✓" : "✗")
    print()

    
    exit(0)
    */
}
