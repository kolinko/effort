//
//  archive.swift
//  effort
//
//  Created 23/03/2024.
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


class Archive : OrderedDict<VectorFloat> {
    var addPrefix : String {
        get {return _addPrefix}
        set (pref){_addPrefix = pref;addIdx=0}
    }
    private var _addPrefix = "idx"
    var addIdx = 0

    func add(prefix pref: String? = nil, _ value: VectorFloat) {
/*
        let valueCopy = value.copy()
        if let pref = pref {
            super.self["\(pref) \(addIdx)"] = valueCopy
        } else {
            super.self["\(addPrefix)\(addIdx)"] = valueCopy
        }
        self.addIdx += 1*/
    }
    func add(prefix pref: String = "idx", _ value: [VectorFloat]) {
        for item in value {
            self.add(item)
        }
    }

    func cosineSimsTo(_ a2: Archive) -> OrderedDict<Float>{
        let out = OrderedDict<Float>()
        for (key, vec) in self {
            out[key] = vec.cosineSimilarityTo(a2[key])
        }
        return out
    }
    
    func strictCompareTo(_ a2: Archive) -> OrderedDict<Bool>{
        let out = OrderedDict<Bool>()
        for (key, vec) in self {
            out[key] = vec.strictCompareTo(a2[key])
        }
        return out
    }
    
}
