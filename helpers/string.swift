//
//  string.swift
//  effort
//
//  Created 17/04/2024.
//

import Foundation


extension String.StringInterpolation {
    mutating func appendInterpolation(perc value: Double) {
        let formatter = NumberFormatter()
        formatter.numberStyle = .percent
        formatter.locale = Locale(identifier: "en_US_POSIX") // Ensures '.' as the decimal separator
        formatter.multiplier = 100
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = 2 // You can adjust precision here as needed
        if let formattedString = formatter.string(for: value) {
            appendLiteral(formattedString)
        }
    }
    
    mutating func appendInterpolation(_ value: Double, precision: Int) {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = Locale(identifier: "en_US_POSIX") // Use a locale with '.' as the decimal separator
        formatter.minimumFractionDigits = precision
        formatter.maximumFractionDigits = precision
        if let formattedString = formatter.string(for: value) {
            appendLiteral(formattedString)
        }
    }
    
    mutating func appendInterpolation(_ value: Float, precision: Int) {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = Locale(identifier: "en_US_POSIX") // Use a locale with '.' as the decimal separator
        formatter.minimumFractionDigits = precision
        formatter.maximumFractionDigits = precision
        if let formattedString = formatter.string(for: value) {
            appendLiteral(formattedString)
        }
    }
    
    mutating func appendInterpolation(_ value: Double, perc: Void) {
            let formatter = NumberFormatter()
            formatter.numberStyle = .percent
            formatter.locale = Locale(identifier: "en_US_POSIX") // Ensures '.' as the decimal separator
            formatter.multiplier = 100
            formatter.minimumFractionDigits = 0
            formatter.maximumFractionDigits = 2 // You can adjust precision here as needed
            if let formattedString = formatter.string(for: value) {
                appendLiteral(formattedString)
            }
        }
}
