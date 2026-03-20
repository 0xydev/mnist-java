import { useRef, useState, useEffect, useCallback } from 'react'
import './App.css'

function App() {
  const canvasRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [training, setTraining] = useState(false)
  const [trainResult, setTrainResult] = useState(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.strokeStyle = '#FFFFFF'
    ctx.lineWidth = 18
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
  }, [])

  const getPos = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height

    if (e.touches) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY,
      }
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    }
  }

  const startDraw = (e) => {
    e.preventDefault()
    const ctx = canvasRef.current.getContext('2d')
    const pos = getPos(e)
    ctx.beginPath()
    ctx.moveTo(pos.x, pos.y)
    setIsDrawing(true)
  }

  const draw = (e) => {
    e.preventDefault()
    if (!isDrawing) return
    const ctx = canvasRef.current.getContext('2d')
    const pos = getPos(e)
    ctx.lineTo(pos.x, pos.y)
    ctx.stroke()
  }

  const stopDraw = (e) => {
    if (e) e.preventDefault()
    setIsDrawing(false)
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.strokeStyle = '#FFFFFF'
    ctx.lineWidth = 18
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    setPrediction(null)
  }

  const predict = useCallback(async () => {
    setLoading(true)
    try {
      const canvas = canvasRef.current
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'))
      const formData = new FormData()
      formData.append('image', blob, 'digit.png')

      const res = await fetch('/api/predict', { method: 'POST', body: formData })
      const data = await res.json()

      if (data.prediction !== undefined) {
        setPrediction(data)
      } else {
        setPrediction({ error: data.message })
      }
    } catch (err) {
      setPrediction({ error: err.message })
    } finally {
      setLoading(false)
    }
  }, [])

  const trainModel = async () => {
    setTraining(true)
    setTrainResult(null)
    try {
      const res = await fetch('/api/train', { method: 'POST' })
      const data = await res.json()
      setTrainResult(data)
    } catch (err) {
      setTrainResult({ status: 'error', message: err.message })
    } finally {
      setTraining(false)
    }
  }

  // Tum 10 rakami (0-9) goster
  const allDigits = prediction && !prediction.error
    ? Array.from({ length: 10 }, (_, i) => {
        const found = prediction.top5.find((item) => item.digit === String(i))
        return { digit: String(i), probability: found ? found.probability : '0.0000' }
      })
    : []

  return (
    <div className="app">
      <h1>MNIST Rakam Tanima</h1>
      <p className="subtitle">Java + DJL + PyTorch</p>

      <div className="main-content">
        <div className="canvas-section">
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            onMouseDown={startDraw}
            onMouseMove={draw}
            onMouseUp={stopDraw}
            onMouseLeave={stopDraw}
            onTouchStart={startDraw}
            onTouchMove={draw}
            onTouchEnd={stopDraw}
          />
          <div className="buttons">
            <button onClick={predict} disabled={loading} className="btn-predict">
              {loading ? 'Tahmin ediliyor...' : 'Tahmin Et'}
            </button>
            <button onClick={clearCanvas} className="btn-clear">Temizle</button>
          </div>

          {/* Debug: Modelin gördüğü 28x28 resim */}
          {prediction?.processedImage && (
            <div className="debug-image">
              <h3>Modelin Gordugu (28x28)</h3>
              <img src={prediction.processedImage} alt="processed" />
            </div>
          )}
        </div>

        <div className="result-section">
          {prediction && !prediction.error && (
            <div className="prediction-result">
              <div className="big-number">{prediction.prediction}</div>
              <div className="confidence">{prediction.confidence} dogruluk</div>
              <div className="all-probs">
                <h3>Tum Olasiliklar</h3>
                {allDigits.map((item) => (
                  <div key={item.digit} className={`prob-row ${item.digit === prediction.prediction ? 'highlight' : ''}`}>
                    <span className="digit-label">{item.digit}</span>
                    <div className="prob-bar-bg">
                      <div
                        className="prob-bar"
                        style={{ width: `${(parseFloat(item.probability) * 100)}%` }}
                      />
                    </div>
                    <span className="prob-value">{(parseFloat(item.probability) * 100).toFixed(2)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {prediction?.error && (
            <div className="error">{prediction.error}</div>
          )}
        </div>
      </div>

      <div className="train-section">
        <button onClick={trainModel} disabled={training} className="btn-train">
          {training ? 'Egitiliyor... (bu biraz surebilir)' : 'Modeli Yeniden Egit'}
        </button>
        {trainResult && (
          <p className={trainResult.status === 'success' ? 'success' : 'error'}>
            {trainResult.message}
          </p>
        )}
      </div>
    </div>
  )
}

export default App
