FROM node:18-alpine

WORKDIR /app

# Install dependencies first for better caching
COPY package*.json ./
RUN npm install

# Copy all the frontend source files
COPY . .

EXPOSE 5173

# Expose port and run Vite on all interfaces
CMD ["npm", "run", "dev", "--", "--host"]
