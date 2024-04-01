//
//  tokeniser.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 25/03/2024.
//

import Foundation

class Tokeniser {
    let data : [String: String]

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
    
    init() {
        let fileUrl = URL(fileURLWithPath: absolutePath + "swift-tokeniser.json")
        let data = try! Data(contentsOf: fileUrl)
        self.data = try! JSONSerialization.jsonObject(with: data, options: []) as! [String:String]
    }
}


