package com.ssafy.ssashinsa.heyfy.fastapi.client;

import com.ssafy.ssashinsa.heyfy.common.exception.CommonErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.fastapi.config.FastApiProperties;
import com.ssafy.ssashinsa.heyfy.fastapi.dto.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

@Slf4j
@Component
@RequiredArgsConstructor
public class FastApiClient {
    private final FastApiProperties fastApiProperties;
    private final WebClient.Builder webClientBuilder;

    public FastApiRateStatusResponseDto getRateStatus(){
        FastApiRateStatusResponseDto response = getClient()
                .get()
                .uri("/push/rate-status")
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            throw new CustomException(CommonErrorCode.INTERNAL_SERVER_ERROR, "Failed to fetch from FastAPI.");
                        }))
                .bodyToMono(FastApiRateStatusResponseDto.class)
                .doOnNext(this::logResponse)
                .block();
        return response;
    }

    public FastApiRealTimeRatesDto getRealTimeRates(){
        FastApiRealTimeRatesDto response = getClient()
                .get()
                .uri("/realtime-rates")
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            throw new CustomException(CommonErrorCode.INTERNAL_SERVER_ERROR, "Failed to fetch from FastAPI.");
                        }))
                .bodyToMono(FastApiRealTimeRatesDto.class)
                .doOnNext(this::logResponse)
                .block();
        return response;
    }
    public FastApiRateGraphDto getRateGraph(){
        FastApiRateGraphDto response = getClient()
                .get()
                .uri("/rate-graph")
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            throw new CustomException(CommonErrorCode.INTERNAL_SERVER_ERROR, "Failed to fetch from FastAPI.");
                        }))
                .bodyToMono(FastApiRateGraphDto.class)
                .doOnNext(this::logResponse)
                .block();
        if(response==null){
            throw new IllegalStateException("Failed to fetch rate-graph from FastAPI.");
        }
        return response;
    }
    public FastApiRateAnalysisDto getRateAnalysis(){
        FastApiRateAnalysisDto response = getClient()
                .get()
                .uri("/rate-analysis")
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            throw new CustomException(CommonErrorCode.INTERNAL_SERVER_ERROR, "Failed to fetch from FastAPI.");
                        }))
                .bodyToMono(FastApiRateAnalysisDto.class)
                .doOnNext(this::logResponse)
                .block();
        return response;
    }

    public FastApiPredictionSummaryResponseDto getPredictionSummary(){
        FastApiPredictionSummaryResponseDto response = getClient()
                .get()
                .uri("/rate-prediction-summary")
                .retrieve()
                .onStatus(HttpStatusCode::isError, r ->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            throw new CustomException(CommonErrorCode.INTERNAL_SERVER_ERROR, "Failed to fetch from FastAPI.");
                        }))
                .bodyToMono(FastApiPredictionSummaryResponseDto.class)
                .doOnNext(this::logResponse)
                .block();
        return response;
    }

    public FastApiTuitionPeriodDto getTuitionPeriod() {
        FastApiTuitionPeriodDto response = getClient()
                .get()
                .uri("/analyze-tuition-period")
                .retrieve()
                .onStatus(HttpStatusCode::isError, r->
                        r.bodyToMono(String.class).flatMap(body -> {
                            log.error("API Error Body: {}", body);
                            throw new CustomException(CommonErrorCode.INTERNAL_SERVER_ERROR, "Faild to fetch from FastAPI.");
                        }))
                .bodyToMono(FastApiTuitionPeriodDto.class)
                .doOnNext(this::logResponse)
                .block();
        return response;
    }

    private WebClient getClient() {
        String baseUrl = fastApiProperties.getFullBaseUrl() + "/api";

        if(baseUrl==null){
            throw new IllegalArgumentException("FastAPI baseUrl is not configured.");
        }
        if (!(baseUrl.startsWith("http://") || baseUrl.startsWith("https://"))) {
            throw new IllegalArgumentException("FastAPI baseUrl must start with http:// or https://. Current value: " + baseUrl);
        }

        return webClientBuilder
                .baseUrl(baseUrl)
                .build();
    }

    private void logRequest(Object requestDto) {
        try {
            log.debug("Request JSON: {}", new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(requestDto));
        } catch (Exception e) {
            log.error("Request logging error", e);
        }
    }

    private void logResponse(Object responseDto) {
        try {
            log.debug("Response JSON: {}", new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(responseDto));
        } catch (Exception e) {
            log.error("Response logging error", e);
        }
    }
}
