package com.vqa.server.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
public class HealthController {

    @Value("${dl.service.url:http://localhost:8000}")
    private String dlServiceUrl;

    @GetMapping("/health")
    public String health() {
        return "Server Backend is running";
    }

    @GetMapping("/call-dl")
    public String callDlService() {
        RestTemplate restTemplate = new RestTemplate();
        try {
            // Gọi endpoint health của DL service
            return restTemplate.getForObject(dlServiceUrl + "/health", String.class);
        } catch (Exception e) {
            return "Failed to reach DL Service: " + e.getMessage();
        }
    }
}