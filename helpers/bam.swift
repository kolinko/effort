//
//  bam.swift
//
//  Periodically reads from buffers to prevent system from throwing them into swap.
//  >30GB - every 1s. <GB - every 100ms, perhaps should be even more intense.

import Foundation

class BufferActivityManager {
    private var dispatchTimer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "com.example.bufferActivityManager", attributes: .concurrent)
    private let gpu = Gpu()
    private var buffers = [Bufferable<Float16>]()
    private var lock = false
    private let bsScalar = ScalarFloat(value: 0.0)
    
    init() {
        if physicalMemoryGB < 30 {
            self.startPeriodicDispatch(interval: 0.1)
        } else {
            self.startPeriodicDispatch(interval: 1.0)

        }
    }

    func addBuffer(_ m: Bufferable<Float16>) {
        lock = true
        buffers.append(m)
        lock = false
    }
        
    func startPeriodicDispatch(interval: TimeInterval = 0.10) {
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
