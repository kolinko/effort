//
//  safetensors.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 03/04/2024.
//

import Foundation

func mergePaths(_ a: String, _ b: String) -> String {
    let basePath = URL(fileURLWithPath: a)
    return basePath.appendingPathComponent(b).path
}

class TensorMetaLoader {
    
    let loaders: [TensorLoader]
    
    init (_ loaders: [TensorLoader]) {
        self.loaders = loaders
    }
    
    
    subscript(index: String) -> MTLBufferable {
            get {
                for loader in loaders {
                    if loader.index.keys.contains(index) {
                        return loader[index]
                    }
                }
                
                preconditionFailure("Tensor not found in library: \(index)")
            }
        }
    
}

class TensorSaver {
    
    var files = [[String: MTLBufferable]]()
    let path : String
    let model : String
    
    init(path: String = "./", model: String = "model") {
        self.path = path
        self.model = model
    }
    
    subscript(index: Int) -> [String: MTLBufferable] {
        get {
            while files.count <= index {
                files.append([String: MTLBufferable]())
            }
            return files[index]
        }
        set (val) {
            while files.count <= index {
                files.append([String: MTLBufferable]())
            }
            files[index] = val
        }

    }
    
    func save() {
        func idToFName(_ id: Int) -> String {
            return String(format:"\(model)-%05d-of-%05d.safetensors", id+1, files.count)
        }
        
        var weightMap = [String: String]()
        
        for id in 0..<self.files.count {
            saveSafetensors(fname: mergePaths(path,idToFName(id)), tensors: files[id])
            for key in files[id].keys {
                weightMap[key] = idToFName(id)
            }
        }
        
        var jsonDict = [String: Any]()
        jsonDict["weight_map"] = weightMap
        let indexData = try! JSONSerialization.data(withJSONObject: jsonDict, options: .prettyPrinted)
        try! indexData.write(to: URL(fileURLWithPath: mergePaths(path, "\(model).safetensors.index.json")))
    }
    
}

class TensorLoader {
    // a library for loading tensors
    // by design it doesn't cache them to not pollute memory
    // classes above it should store references
    
    let index: [String: String]
    let path: String
    
    static func loadVec(_ fname: String) -> VectorFloat {
        let tl = TensorLoader(path: "./", model: fname)
        return tl["h"] as! VectorFloat
    }
    
    init() {
        self.path = "/dev/null"
        self.index = [String: String]()
    }
    
    init(path: String = "./", model: String = "model") {
        self.path = path
        
        let fileUrl = URL(fileURLWithPath: mergePaths(path, "\(model).safetensors.index.json"))
        let data = try! Data(contentsOf: fileUrl)
        self.index = (try! JSONSerialization.jsonObject(with: data, options: []) as! [String:Any])["weight_map"]! as! [String: String]
    }
    
    func vector(_ index: String, assertShape: [Int]? = nil) -> Vector {
        let out = self[index] as! Vector
        assert(assertShape == nil || out.shape == assertShape, "wrong shape loaded! \(index) has shape \(out.shape), should be \(assertShape!)")
        return out
    }

    func matrix(_ index: String, assertShape: [Int]? = nil) -> Matrix {
        let out = self[index] as! Matrix
        assert(assertShape == nil || out.shape == assertShape, "wrong shape loaded! \(index) has shape \(out.shape), should be \(assertShape!)")
        return out
    }

    
    subscript(index: String) -> MTLBufferable {
            get {
               return fetchTensor(keyname: index)
            }
        }
    
    func fetchTensor(keyname _keyname: String) -> MTLBufferable {
        //var tensors = [String: MTLBufferable]()
        
        var fname = ""
        var keyname = ""
        if self.index.keys.contains(_keyname) {
            keyname = _keyname
        } else {
            keyname = String(_keyname.dropLast(4))
        }
        
//        if keyname.contains(".bucket.stats") {
//            keyname = String(_keyname.dropLast(17)) + ".stats"

//        }
        
        precondition(self.index.keys.contains(keyname), "\(keyname) not found in the safetensors lib!")
        
        fname = mergePaths(self.path, self.index[keyname]!)
        
        guard let fileHandle = FileHandle(forReadingAtPath: fname) else {
            print("Failed to open file \(fname).")
            exit(1)
        }
    
       // print("Loading \(keyname) from \(fname)...")
        // Read the first 8 bytes to get the header size
        let headerSizeData = fileHandle.readData(ofLength: 8)
        let headerSize = UInt64(littleEndian: headerSizeData.withUnsafeBytes { $0.load(as: UInt64.self) })
        
        //            print("headerSize \(headerSize)")
        // Read the header
        let headerData = fileHandle.readData(ofLength: Int(headerSize))
        let headerJson = try! JSONSerialization.jsonObject(with: headerData) as! [String: Any]
        let value = headerJson[keyname]!
        // Parse tensors info from the header
        let tensorDict = value as! [String: Any]
        let dtype = tensorDict["dtype"] as! String
        var numBytes = 2
        if dtype == "F32" {
            numBytes = 4
        }
        
        precondition(["BF16", "F16", "F32"].contains(dtype), "number type in safetensors unsupported. got \(dtype), expected either BF16 or F16. Did you try to load a effortized model?")
        
        let shape = tensorDict["shape"] as! [Int]
        var offsets = tensorDict["data_offsets"] as! [UInt64]
        precondition(offsets.count == 2)
        precondition(offsets[1]-offsets[0] == shape.reduce(1, *)*numBytes)
        offsets[0]+=headerSize+8
        
        let _buffer = loadBinarySegment(named: fname, fromOffset: off_t(offsets[0]), withShape: shape, bytesPerEl: numBytes)
        let buffer = _buffer!
        var out : MTLBufferable
        
        if dtype == "F32" {
            if shape.count == 1 {
                out = VectorFloat(shape: shape, buffer: buffer)
            } else if shape.count == 2 {
                out = MatrixFloat(shape: shape, buffer: buffer)
            } else {
                preconditionFailure("unsupported shape for F32")
            }
//            out = MatrixFloat(shape: shape, buffer: buffer)
        } else if shape.count == 1 {
            out = Vector(shape: shape, buffer: buffer)
        } else if shape.count == 2 {
            out = Matrix(shape: shape, buffer: buffer)
        } else if shape.count >= 3 {
            print("WARN: tensor has too many dims. Loading anyway and hoping for the best. \(keyname) is \(shape)")
            out = Matrix3D(shape: shape, buffer: buffer)
        } else {
            preconditionFailure("unknown error while loading tensors")
        }
        
        if dtype == "BF16" {
            out.convertBF16()
            gpu.eval()
        }
        
        fileHandle.closeFile()
        
        return out

    }
    
}



func saveSafetensors(fname filePath: String, tensors: [String: MTLBufferable]) {
    FileManager.default.createFile(atPath: filePath, contents: nil, attributes: nil)
       guard let fileHandle = FileHandle(forWritingAtPath: filePath) else {
           print("Failed to open file \(filePath) for writing.")
           return
       }
       
    do {
        // Construct the header dictionary
        var headerDict = [String: Any]()
        var currOffset : UInt64 = 0
        var tensorsArray = [MTLBufferable]()
        
        for (key, val) in tensors {
            tensorsArray.append(val)
            
            if let v = val as? Bufferable<Float16> {
                headerDict[key] = [
                    "dtype": "F16",
                    "shape": v.shape,
                    "data_offsets": [currOffset, currOffset + UInt64(v.countBytes)]
                ]
                currOffset += UInt64(v.countBytes)
            } else if let v = val as? Bufferable<Float> {
                headerDict[key] = [
                    "dtype": "F32",
                    "shape": v.shape,
                    "data_offsets": [currOffset, currOffset + UInt64(v.countBytes)]
                ]
                currOffset += UInt64(v.countBytes)
            } else {
                preconditionFailure("unknown bufferable type when serializing \(filePath)")
            }
        }
        headerDict["__metadata__"] = ["description": "Bucket weights format, see mixtral-kolinko at github"]

        let headerData = try JSONSerialization.data(withJSONObject: headerDict)
        let headerSize = UInt64(headerData.count)
        
        var leHeaderSize = headerSize.littleEndian
        let headerSizeData = Data(bytes: &leHeaderSize, count: MemoryLayout<UInt64>.size)
        fileHandle.write(headerSizeData)
        fileHandle.write(headerData)

        
        // Write each tensor's data
        for tensor in tensorsArray {
            let data = Data(bytes: tensor.buffer.contents(), count: tensor.buffer.length)//, deallocator: .none)
                // Assuming the data is already in BF16 format; otherwise, you would need to convert it.
            fileHandle.write(data)
        }

    } catch {
            print("An error occurred while writing \(filePath): \(error)")
    }
 
    fileHandle.closeFile()
    print("saved \(filePath)")
}

extension Data {
    init<T>(from value: T) {
        self = Swift.withUnsafeBytes(of: value) { Data($0) }
    }
}


import Metal

func alignOffsetToPageSize(offset: off_t) -> (alignedOffset: off_t, offsetAdjustment: off_t) {
    let pageSize = off_t(getpagesize())
    let offsetAdjustment = offset % pageSize
    let alignedOffset = offset - offsetAdjustment
    return (alignedOffset, offsetAdjustment)
}


func loadBinarySegment(named fileName: String, fromOffset offset: off_t, withShape shape: [Int], bytesPerEl: Int) -> MTLBuffer? {
    let device = gpu.device
    let fileURL = URL(fileURLWithPath: fileName)

    // Calculate the expected size based on the shape, assuming Float16 (2 bytes)
    let expectedCount = shape.reduce(1, *)
    let expectedSize = expectedCount * bytesPerEl

    // Open the file
    let fileDescriptor = open(fileURL.path, O_RDONLY)
    precondition(fileDescriptor != -1, "Cannot open file \(fileName).")

    // Seek to the specified offset
    let seekResult = lseek(fileDescriptor, offset, SEEK_SET)
    precondition(seekResult != -1, "Seeking failed.")

    // Calculate aligned offset for mmap
    let (alignedOffset, offsetAdjustment) = alignOffsetToPageSize(offset: offset)

    // Memory map the file segment, using alignedOffset
    let actualSize = off_t(expectedSize)+offsetAdjustment
    guard let dataPointer = mmap(nil, Int(actualSize), PROT_READ, MAP_PRIVATE, fileDescriptor, alignedOffset) else {
        preconditionFailure("Memory mapping of \(fileName) failed.")
    }

    // Adjust dataPointer by offsetAdjustment to get the actual start of your data
    let actualDataPointer = dataPointer.advanced(by: Int(offsetAdjustment))
    
    guard let buffer = device.makeBuffer(bytes: actualDataPointer, length: expectedSize, options: .storageModeShared) else {
        close(fileDescriptor)
        preconditionFailure("cannot load the buffer: \(offset), \(alignedOffset)")
    }

    close(fileDescriptor)
    return buffer
}


/*
func loadBinarySegment(named fileName: String, fromOffset offset: off_t, withShape shape: [Int], bytesPerEl: Int) -> (Int, MTLBuffer?) {
    let fileURL = URL(fileURLWithPath: fileName)

    // Calculate the expected size based on the shape
    let expectedCount = shape.reduce(1, *)
    let expectedSize = expectedCount * bytesPerEl

    do {
        // Read file data
        let fileData = try Data(contentsOf: fileURL)
        
        // Check if the offset is within the file data bounds
        guard fileData.count > offset else {
            print("Offset is beyond the file size.")
            return (0, nil)
        }
        
        // Create a buffer for the expected data size
        guard let buffer = gpu.device.makeBuffer(length: expectedSize, options: .storageModeShared) else {
            print("Failed to create MTLBuffer.")
            return (0, nil)
        }
        
        // Copy data from the specified offset into the buffer
                fileData[Int(offset)...].prefix(expectedSize).withUnsafeBytes { rawBufferPointer in
                    if let baseAddress = rawBufferPointer.baseAddress {
                        buffer.contents().copyMemory(from: baseAddress, byteCount: expectedSize)
                    }
                }
        
        return (0, buffer)
    } catch {
        print("Failed to read file data: \(error)")
        return (0, nil)
    }
}
*/
