//
//  watchman.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 29/03/2024.
//

import Foundation

class BufferActivityManager {
    private var dispatchTimer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "com.example.bufferActivityManager", attributes: .concurrent)
    private let gpu = Gpu()
    private var buffers = [Bufferable<Float16>]()
    private var lock = false
    private let bsScalar = ScalarFloat(value: 0.0)
    
    init() {}

    func addBuffer(_ m: Bufferable<Float16>) {
        lock = true
        buffers.append(m)
        lock = false
    }
        
    func startPeriodicDispatch(interval: TimeInterval = 0.1) {
        stopPeriodicDispatch() // Stop any existing timer

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: interval)
        timer.setEventHandler { [weak self] in
            self?.dispatch()
        }
        timer.resume()
        dispatchTimer = timer
    }

    func stopPeriodicDispatch() {
        dispatchTimer?.cancel()
        dispatchTimer = nil
    }

    private func dispatch() {
        // Your dispatch function that executes the kernel
        // Assume this function exists and dispatches the Metal compute kernel
        //print("Dispatching compute kernel...")
        if !lock {
            lock = true
            for b in buffers {
                gpu.deploy("touch", buffers: [b, bsScalar], ints: [b.count], threadCount: 32)
            }
            lock = false
            gpu.eval()

        }
    }

    deinit {
        stopPeriodicDispatch()
    }
}
