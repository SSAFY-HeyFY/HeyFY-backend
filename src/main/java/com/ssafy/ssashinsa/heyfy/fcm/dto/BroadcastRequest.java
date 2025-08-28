package com.ssafy.ssashinsa.heyfy.fcm.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class BroadcastRequest {
    private String title;
    private String body;
}