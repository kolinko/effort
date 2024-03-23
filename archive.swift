//
//  archive.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 23/03/2024.
//

import Foundation

class Archive : Sequence{
    var data = [String: Vector]()
    var order = [String]()
    
    subscript(index: String) -> Vector {
        get { data[index]! }
        set {
            data[index] = newValue
            order.append(index)
        }
    }

    func makeIterator() -> AnyIterator<(String, Vector)> {
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
