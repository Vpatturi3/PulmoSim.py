import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import Home from './pages/Home'
import Results from './pages/Results'
import DemoViewer from './pages/DemoViewer'
import './styles.css'

function RouteError() {
  return (
    <div style={{ padding: 24 }}>
      <h2>Something went wrong</h2>
      <p>Try reloading the page or going back.</p>
      <a className="btn" href="/">Back to Home</a>
    </div>
  )
}

const router = createBrowserRouter([
  { path: '/', element: <Home />, errorElement: <RouteError /> },
  { path: '/demo-viewer', element: <DemoViewer />, errorElement: <RouteError /> },
  { path: '/results', element: <Results />, errorElement: <RouteError /> },
])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)


