//
//  archive.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 23/03/2024.
//

import Foundation

class OrderedDict<Value>: Sequence {
    var data = [String: Value]()
    var order = [String]()
    
    subscript(index: String) -> Value {
        get { return data[index]! }
        set {
            if data[index] == nil {
                order.append(index)
            }
            data[index] = newValue
        }
    }
    
    func makeIterator() -> AnyIterator<(String, Value)> {
        var currentIndex = 0
        return AnyIterator {
            guard currentIndex < self.order.count else {
                return nil
            }
            let key = self.order[currentIndex]
            guard let value = self.data[key] else {
                return nil
            }
            currentIndex += 1
            return (key, value)
        }
    }
}

class Archive : OrderedDict<Vector> {

    func cosineSimsTo(_ a2: Archive) -> OrderedDict<Float>{
        let out = OrderedDict<Float>()
        for (key, vec) in self {
            out[key] = vec.cosineSimilarityTo(a2[key])[0]
        }
        return out
    }
    
}
