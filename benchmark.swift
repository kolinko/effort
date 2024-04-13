//
//  benchmark.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 13/04/2024.
//

import Foundation

func runBenchmark() {
    let query = "[INST]Write an extremely long and convoluted story about potatoes[/INST]"
    //Write a condensed perl implementations of the following: Dijkstra's algorithm, text search and quicksort. No comments, just write the code. Include data loaders.
    let tokIds = encode(prompt: query)
    let baseline = runNetwork(isTest: true, tokens: t.embed(tokIds), effort: 1, srcTokenIds: tokIds)

    let tokenIds2 = baseline.hitMiss
    let control = runNetwork(isTest: true, tokens: t.embed(tokenIds2), effort: 1).hitMiss
    
    for effort in [1, 0.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02] {
        let test = runNetwork(isTest: true, tokens: t.embed(tokenIds2), effort: effort, srcTokenIds: tokenIds2).hitMiss
        var num = 0
        for i in 0..<control.count {
            num += control[i] == test[i] ? 1 : 0
        }
        let res = Double(100*Float(num)/Float(control.count))
        print("\(Int(effort*100))% -> " + "\(res, precision: 2)%")
        
    }
  
    exit(0)
}
