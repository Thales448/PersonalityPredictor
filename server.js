const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from the public folder
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Endpoint to receive survey responses
app.post('/submit', (req, res) => {
  const answers = req.body;
  console.log('Received survey answers:', answers);

  // TODO: Convert answers to Big Five scores
  // TODO: Match against characters
  // Dummy response for now
  res.json({
    message: 'Survey received!',
    match: 'TBD',
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
