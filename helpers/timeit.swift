//
//  profile.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 23/03/2024.
//

import Foundation

func timeIt(repeats: Int = 10000, multiplier: Double = 1.0, brief: Bool = true, _ closure: (Int) -> Void) {
    if !brief {
        print("profiling, reps \(repeats)...")
    }
    var goTime = Date()
    // warmup loop & in case of capture - just the roop
    for i in 0..<1000 {
        closure(i)
    }
    gpu.eval()
    
    for i in 0..<repeats {
        closure(i)
    }
    goTime = Date()
    gpu.eval()
    let epl = Date().timeIntervalSince(goTime)*1000/Double(repeats)
    if (!brief) {
        print("persec \(Double(repeats) / Date().timeIntervalSince(goTime), precision: 2) runs")
        print()
    }

    // * num experts * num big muls (kv,kq.. sum to one, then w1,w2,w3) * num_layers
    print("tpt \(epl*1*4*32*multiplier, precision: 2) ms")
    print("spd \(1000/(epl*1*4*32*multiplier), precision: 2) tps")
}
