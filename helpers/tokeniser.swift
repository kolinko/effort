//
//  tokeniser.swift
//  effort
//

import Foundation


// This whole module should be refactored into tokeniser2, and swift-tokeniser.json dependency
// removed altogether

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
        self.data = loadJson("./swift-tokeniser.json") as! [String:String]
        self.tokEmbeddings = modelData.tokEmbeddings.asVectorList()
    }

    func embed(_ prompt: String) -> [VectorFloat] {
        return embed(encode(prompt: prompt))
    }
    
    func embed(_ tokIds: [Int]) -> [VectorFloat] {
        var tokens = [VectorFloat]()
        
        for t in tokIds {
            tokens.append(self.tokEmbeddings[t].asFloat32())
        }
        
        return tokens
    }
}


