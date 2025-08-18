package com.ssafy.ssashinsa.heyfy.log.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.zalando.logbook.BodyFilter;

import java.util.Set;

public class CustomBodyFilter implements BodyFilter {

    private final Set<String> keysToRedact;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public CustomBodyFilter(Set<String> keysToRedact) {
        this.keysToRedact = keysToRedact;
    }

    @Override
    public String filter(String contentType, String body) {
        if (contentType == null || !contentType.contains("application/json") || body == null) {
            return body;
        }
        try {
            JsonNode root = objectMapper.readTree(body);
            maskFields(root);
            return objectMapper.writeValueAsString(root);
        } catch (Exception e) {
            // If parsing fails, return the original body
            return body;
        }
    }

    private void maskFields(JsonNode node) {
        if (node.isObject()) {
            ObjectNode obj = (ObjectNode) node;
            obj.fieldNames().forEachRemaining(field -> {
                if (keysToRedact.contains(field)) {
                    obj.put(field, "***");
                } else {
                    maskFields(obj.get(field));
                }
            });
        } else if (node.isArray()) {
            for (JsonNode item : node) {
                maskFields(item);
            }
        }
    }
}