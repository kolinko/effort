//
//  tokeniser.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 25/03/2024.
//

import Foundation

class Tokeniser {
    let data : [String: String]
    let tokEmbeddings: [Vector]
    
    subscript(token: Int) -> String {
        return data[String(token)]!
    }

    func decode(_ token: Int) -> String {
        return data[String(token)]!
    }
    
    func decode(_ tokens: [Int], delim: String = ";") -> String {
        var out = ""
        for token in tokens {
            out += data[String(token)]! + delim
        }
        return out
    }
    
    init(_ modelData: Model) {
        let fileUrl = URL(fileURLWithPath: absolutePath + "swift-tokeniser.json")
        let data = try! Data(contentsOf: fileUrl)
        self.data = try! JSONSerialization.jsonObject(with: data, options: []) as! [String:String]
        self.tokEmbeddings = modelData.tokEmbeddings.asVectorList()
    }

    func embed(_ prompt: String) -> [VectorFloat] {
        return embed(encode(prompt: prompt))
    }
    
    func embed(_ tokIds: [Int]) -> [VectorFloat] {
        var tokens = [VectorFloat]()
        
        let tokEmbeddings = modelData.tokEmbeddings.asVectorList()
        for t in tokIds {
            tokens.append(tokEmbeddings[t].asFloat32())
        }
        
        return tokens
    }
}


