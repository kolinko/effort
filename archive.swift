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
    var addPrefix : String {
        get {return _addPrefix}
        set (pref){_addPrefix = pref;addIdx=0}
    }
    private var _addPrefix = "idx"
    var addIdx = 0

    func add(prefix pref: String? = nil, _ value: Vector, seriously: Bool = false) {
        return
        let valueCopy = value.copy()
        if let pref = pref {
            super.self["\(pref) \(addIdx)"] = valueCopy
        } else {
            super.self["\(addPrefix)\(addIdx)"] = valueCopy
        }
        self.addIdx += 1
    }
    func add(prefix pref: String = "idx", _ value: [Vector], seriously: Bool = false) {
        for item in value {
            self.add(item, seriously: seriously)
        }
    }

    
    func cosineSimsTo(_ a2: Archive) -> OrderedDict<Float>{
        let out = OrderedDict<Float>()
        for (key, vec) in self {
            out[key] = vec.cosineSimilarityTo(a2[key])[0]
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

    // Serialization of the Archive
    func serialize(fname: String) {
        let metadataPath = "archive/\(fname).json"
        let dataPath = "archive/\(fname).bin"

        var metadata = [String: Any]()
        var currentOffset = 0
        
        // Open or create the binary file for writing
        let dataFileURL = URL(fileURLWithPath: dataPath)
        guard let dataFileHandle = try? FileHandle(forWritingTo: dataFileURL) else {
            print("Cannot open data file for writing.")
            return
        }
        
        for (key, vector) in self {
            let dataSize = vector.shape.reduce(1, *) * MemoryLayout<Float16>.size
            
            // Prepare the entry for the metadata
            metadata[key] = ["shape": vector.shape, "offset": currentOffset, "size": dataSize]
            
            // Assuming vector.buffer.contents() gives us the pointer to the data
            let bufferPointer = vector.buffer.contents().assumingMemoryBound(to: UInt8.self)
            
            // Write the vector data to the file
            let data = Data(bytes: bufferPointer + vector.offset, count: dataSize)
            if dataFileHandle.seekToEndOfFile() == UInt64(currentOffset) {
                dataFileHandle.write(data)
            } else {
                print("Error seeking to correct offset in data file.")
                return
            }
            
            // Update the offset for the next vector
            currentOffset += dataSize
        }
        
        // Close the data file
        dataFileHandle.closeFile()
        
        // Convert metadata dictionary to Data and write to the metadata file
        if let jsonData = try? JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted]) {
            try? jsonData.write(to: URL(fileURLWithPath: metadataPath))
        } else {
            print("Error serializing metadata.")
        }
    }
    
}
