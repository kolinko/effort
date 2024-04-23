//
//  helpers.swift
//  effort
//
//  Created by Tomasz Kolinko on 23/04/2024.
//

import Foundation


func loadJson(_ path: String) -> Any {
    let fileUrl = URL(fileURLWithPath: path)
    let data = try! Data(contentsOf: fileUrl)
    return try! JSONSerialization.jsonObject(with: data, options: [])

}
