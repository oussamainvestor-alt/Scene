const http = require('http')
const fs = require('fs')
const path = require('path')

const LOG_DIR = '/home/ubuntu/Logs'
const LOG_FILE = path.join(LOG_DIR, 'video-debug.log')

fs.mkdirSync(LOG_DIR, { recursive: true })
// Clear log on server start so each session is fresh
fs.writeFileSync(LOG_FILE, `=== Log server started ${new Date().toISOString()} ===\n`)

const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

  if (req.method === 'OPTIONS') {
    res.writeHead(204)
    res.end()
    return
  }

  if (req.method === 'POST' && req.url === '/log') {
    let body = ''
    req.on('data', chunk => { body += chunk })
    req.on('end', () => {
      const line = `[${new Date().toISOString()}] ${body}\n`
      fs.appendFileSync(LOG_FILE, line)
      process.stdout.write(line)
      res.writeHead(200)
      res.end()
    })
    return
  }

  res.writeHead(404)
  res.end()
})

server.listen(3001, () => {
  console.log(`Log server → ${LOG_FILE}`)
})
