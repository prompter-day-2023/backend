## 꾸미 Server Docker 동작 방법 ✨
1. Docker build
   
   ```
   docker build -t ggoomi-backend  .
   ```

2. Docker run
   
    ```
      docker run -it -p 5123:5123 --rm --name ggoomi-backend ggoomi-backend
    ```