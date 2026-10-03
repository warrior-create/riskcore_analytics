Attribute VB_Name = "VaRCalculator"
'============================================================
' MarketRisk-Lab: Legacy VBA VaR Calculator (v1 - baseline)
' This file represents the LEGACY TOOL being migrated to Python.
' DO NOT USE FOR PRODUCTION — superseded by Python implementation.
'============================================================
' Assumptions (hardcoded):
'   - 250-day lookback
'   - Equal portfolio weights
'   - Simple (not EWMA) covariance
'   - Normal distribution
'   - 99% one-tailed VaR
'============================================================
Option Explicit

Const LOOKBACK_DAYS As Integer = 250
Const CONFIDENCE As Double = 0.99

' -------------------------------------------------------
' Main entry point: compute VaR from price sheet
' Called from the "Calculate VaR" button in the workbook
' -------------------------------------------------------
Sub CalculateVaR()
    Dim ws As Worksheet
    Dim priceRange As Range
    Dim nTickers As Integer
    Dim nDays As Integer
    Dim returns() As Double
    Dim covMatrix() As Double
    Dim weights() As Double
    Dim portVol As Double
    Dim varResult As Double
    Dim esResult As Double
    
    Set ws = ThisWorkbook.Sheets("Prices")
    
    ' Read dimensions
    nTickers = ws.Range("B1").End(xlToRight).Column - 1
    nDays = ws.Range("A2").End(xlDown).Row - 1
    
    If nDays < LOOKBACK_DAYS Then
        MsgBox "Insufficient data: " & nDays & " days (need " & LOOKBACK_DAYS & ")"
        Exit Sub
    End If
    
    ' Build equal weights
    ReDim weights(1 To nTickers)
    Dim i As Integer
    For i = 1 To nTickers
        weights(i) = 1.0 / nTickers
    Next i
    
    ' Compute log returns (last LOOKBACK_DAYS)
    ReDim returns(1 To LOOKBACK_DAYS - 1, 1 To nTickers)
    Dim startRow As Long
    startRow = nDays - LOOKBACK_DAYS + 2   ' 1-indexed, row 2 = first data
    
    Dim t As Long, j As Integer
    Dim prevPrice As Double, currPrice As Double
    
    For t = 1 To LOOKBACK_DAYS - 1
        For j = 1 To nTickers
            prevPrice = ws.Cells(startRow + t - 1, j + 1).Value
            currPrice = ws.Cells(startRow + t, j + 1).Value
            If prevPrice > 0 Then
                returns(t, j) = Log(currPrice / prevPrice)
            Else
                returns(t, j) = 0
            End If
        Next j
    Next t
    
    ' Compute simple covariance matrix
    ReDim covMatrix(1 To nTickers, 1 To nTickers)
    Dim means() As Double
    ReDim means(1 To nTickers)
    
    For j = 1 To nTickers
        Dim total As Double
        total = 0
        For t = 1 To LOOKBACK_DAYS - 1
            total = total + returns(t, j)
        Next t
        means(j) = total / (LOOKBACK_DAYS - 1)
    Next j
    
    Dim k As Integer
    For i = 1 To nTickers
        For k = 1 To nTickers
            Dim cov As Double
            cov = 0
            For t = 1 To LOOKBACK_DAYS - 1
                cov = cov + (returns(t, i) - means(i)) * (returns(t, k) - means(k))
            Next t
            covMatrix(i, k) = cov / (LOOKBACK_DAYS - 2)
        Next k
    Next i
    
    ' Portfolio variance = w' * Sigma * w
    Dim portVariance As Double
    portVariance = 0
    For i = 1 To nTickers
        For j = 1 To nTickers
            portVariance = portVariance + weights(i) * covMatrix(i, j) * weights(j)
        Next j
    Next i
    
    portVol = Sqr(portVariance)
    
    ' VaR = z * sigma (z = NormInv(0.99) = 2.3263)
    Dim z As Double
    z = Application.NormInv(CONFIDENCE, 0, 1)
    varResult = portVol * z
    
    ' ES = sigma * phi(z) / (1 - alpha)
    esResult = portVol * Application.NormDist(z, 0, 1, False) / (1 - CONFIDENCE)
    
    ' Write results to Output sheet
    Dim wsOut As Worksheet
    Set wsOut = ThisWorkbook.Sheets("Results")
    wsOut.Range("B2").Value = "Model"
    wsOut.Range("C2").Value = "Legacy VBA Parametric"
    wsOut.Range("B3").Value = "Confidence"
    wsOut.Range("C3").Value = CONFIDENCE
    wsOut.Range("B4").Value = "Lookback Days"
    wsOut.Range("C4").Value = LOOKBACK_DAYS
    wsOut.Range("B5").Value = "Portfolio Volatility"
    wsOut.Range("C5").Value = portVol
    wsOut.Range("B6").Value = "VaR (1-day)"
    wsOut.Range("C6").Value = varResult
    wsOut.Range("B7").Value = "ES (1-day)"
    wsOut.Range("C7").Value = esResult
    wsOut.Range("B8").Value = "Computed At"
    wsOut.Range("C8").Value = Now()
    
    MsgBox "VaR calculated: " & Format(varResult, "0.0000") & _
           " | ES: " & Format(esResult, "0.0000"), vbInformation
End Sub

' -------------------------------------------------------
' Exception counter — loops over history, counts breaches
' -------------------------------------------------------
Sub CountExceptions()
    Dim ws As Worksheet
    Dim wsPnl As Worksheet
    Set ws = ThisWorkbook.Sheets("VaRHistory")
    Set wsPnl = ThisWorkbook.Sheets("PnL")
    
    Dim nRows As Long
    nRows = ws.Range("A2").End(xlDown).Row - 1
    
    Dim exceptions As Integer
    exceptions = 0
    
    Dim t As Long
    For t = 1 To nRows
        Dim varEst As Double
        Dim actualLoss As Double
        varEst = ws.Cells(t + 1, 2).Value
        actualLoss = -wsPnl.Cells(t + 1, 2).Value   ' P&L is negative when loss
        If actualLoss > varEst Then
            exceptions = exceptions + 1
            ws.Cells(t + 1, 3).Value = "EXCEPTION"
            ws.Cells(t + 1, 3).Interior.Color = RGB(255, 0, 0)
        Else
            ws.Cells(t + 1, 3).Value = ""
        End If
    Next t
    
    ' Traffic light
    Dim lastRow As Long
    lastRow = ws.Range("A2").End(xlDown).Row + 2
    ws.Cells(lastRow, 1).Value = "Exception Count (250d):"
    ws.Cells(lastRow, 2).Value = exceptions
    
    Dim trafficLight As String
    If exceptions <= 4 Then
        trafficLight = "GREEN"
        ws.Cells(lastRow, 3).Interior.Color = RGB(0, 255, 0)
    ElseIf exceptions <= 9 Then
        trafficLight = "AMBER"
        ws.Cells(lastRow, 3).Interior.Color = RGB(255, 165, 0)
    Else
        trafficLight = "RED"
        ws.Cells(lastRow, 3).Interior.Color = RGB(255, 0, 0)
    End If
    ws.Cells(lastRow, 3).Value = trafficLight
    
    MsgBox "Exceptions in last 250 days: " & exceptions & " — " & trafficLight
End Sub
