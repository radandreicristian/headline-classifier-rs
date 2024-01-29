FROM rust:latest as build

WORKDIR /app

COPY . .

RUN cargo build --release --bin inference

# Release
FROM debian:stable-slim as release

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=build /app/target/release/inference .

# Expose the port on which your HTTP API will listen
EXPOSE 3030

# Specify the command to run when the container starts
CMD ["./inference"]