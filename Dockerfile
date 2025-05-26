# Use Node.js base image
FROM node:18-alpine

# Create app directory
WORKDIR /app

# Install app dependencies
COPY package*.json ./
RUN npm install

# Copy app source
COPY . .

# Expose port (use same as in server.js)
EXPOSE 3000

# Start server
CMD ["node", "server.js"]
